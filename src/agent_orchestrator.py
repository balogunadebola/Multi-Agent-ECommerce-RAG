"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime, timezone

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from agent_observability import tool, tracer, trace_kb_retrieval
# AgentCore CLI naming compatibility with the official Udacity starter config.
config.AGENTCORE_PROJECT_NAME = getattr(
    config, "AGENTCORE_PROJECT_NAME", "udacity"
)
config.AGENTCORE_AGENT_NAME = getattr(
    config, "AGENTCORE_AGENT_NAME", "agentcore_runtime"
)
config.AGENTCORE_RUNTIME_NAME = getattr(
    config,
    "AGENTCORE_RUNTIME_NAME",
    f"{config.AGENTCORE_PROJECT_NAME}_{config.AGENTCORE_AGENT_NAME}",
)

import agentcore_cli
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════

def _apply_guardrail() -> dict:
    """Return guardrail settings for BedrockModel from the runtime environment."""
    guardrail_id = os.environ.get("GUARDRAIL_ID", config.GUARDRAIL_ID)
    guardrail_version = os.environ.get(
        "GUARDRAIL_VERSION", config.GUARDRAIL_VERSION
    )

    if not guardrail_id or not guardrail_version.isdigit():
        raise ValueError(
            "A guardrail ID and numbered guardrail version are required."
        )

    return {
        "guardrail_id": guardrail_id,
        "guardrail_version": guardrail_version,
        "guardrail_redact_input": True,
        "guardrail_redact_input_message": (
            "I'm sorry, but I can't help with that request. "
            "Please ask a NovaMart customer-support question."
        ),
        "guardrail_redact_output": True,
        "guardrail_redact_output_message": (
            "I'm sorry, but I can't provide that response. "
            "I can still help with an appropriate NovaMart support request."
        ),
    }

# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────


def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
        **_apply_guardrail(),
    )

    system_prompt = """
    You are NovaMart's Inventory Agent.

    Your responsibility is to retrieve accurate customer and order facts
    using your available tools.

    Rules:
    - Use the tools to look up customer and order information.
    - Never invent customer records, orders, dates, prices, or statuses.
    - If a record is not found, clearly report that it was not found.
    - Report the facts you retrieve in a clear, structured way.
    - Do not decide whether a return or refund should be approved.
    - Do not initiate refunds or modify order records.
    - Do not answer policy questions from memory; retrieve facts only.
    """

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Retrieve the details and current status of an order.

        Args:
            customer_id: The customer's unique identifier.
            order_id: The unique order identifier.

        Returns:
            The order record, or an error if the order does not exist.
        """
        orders_table = dynamodb.Table(config.ORDERS_TABLE)

        response = orders_table.get_item(
            Key={
                "customer_id": customer_id,
                "order_id": order_id,
            }
        )
        order = response.get("Item")

        if not order:
            return {
                "found": False,
                "customer_id": customer_id,
                "order_id": order_id,
                "error": "Order not found",
            }

        return {
            "found": True,
            "order": order,
        }

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier and profile details.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            Customer profile including tier and account details.
        """
        customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

        response = customers_table.get_item(
            Key={"customer_id": customer_id}
        )
        customer = response.get("Item")

        if not customer:
            return {
                "found": False,
                "customer_id": customer_id,
                "error": "Customer not found",
            }

        # Return only relevant profile fields.
        return {
            "found": True,
            "customer_id": customer.get("customer_id"),
            "name": customer.get("name"),
            "tier": customer.get("tier"),
            "account_created": customer.get("account_created"),
            "total_orders": customer.get("total_orders"),
            "preferred_contact": customer.get("preferred_contact"),
        }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders belonging to a customer.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            The customer's orders, sorted newest first.
        """
        from boto3.dynamodb.conditions import Attr

        orders_table = dynamodb.Table(config.ORDERS_TABLE)

        # A scan with a filter is used here because the seed data
        # does not establish a customer_id index on the orders table.
        response = orders_table.scan(
            FilterExpression=Attr("customer_id").eq(customer_id)
        )
        orders = response.get("Items", [])

        # Continue reading if DynamoDB paginates the scan.
        while "LastEvaluatedKey" in response:
            response = orders_table.scan(
                FilterExpression=Attr("customer_id").eq(customer_id),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            orders.extend(response.get("Items", []))

        orders.sort(
            key=lambda order: order.get("order_date", ""),
            reverse=True,
        )

        return {
            "customer_id": customer_id,
            "count": len(orders),
            "orders": orders,
        }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────


def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    refund_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
        **_apply_guardrail(),
    )

    refund_system_prompt = """
You are the NovaMart Refund Agent.

Your job is to evaluate return requests and initiate eligible returns.

Rules:
1. Use get_inventory_context to inspect the inventory findings recorded in
   WorkflowState before making a decision.
2. Verify order ownership, order date, customer tier, and any available
   return-eligibility flag before initiating a return.
3. The Standard return window is 30 days. The Premium return window is
   60 days.
4. Do not claim a return was initiated unless initiate_refund confirms it.
5. Do not invent order facts or assume that a missing field means an order
   is eligible.
6. If information is missing or validation fails, explain what needs to
   be checked.
7. Distinguish a return request from a completed refund. Initiating a return
   does not mean money has already been refunded.
"""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read inventory findings from the session's WorkflowState.

        Args:
            session_id: Identifier of the customer-support session.

        Returns:
            A dictionary containing success status and inventory findings,
            or an error message if the state is unavailable.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return {
                "success": False,
                "message": "No workflow state was found for this session.",
                "inventory_agent": {},
            }

        return {
            "success": True,
            "inventory_agent": state.get("inventory_agent", {}),
        }

    @tool
    def initiate_refund(
        customer_id: str,
        order_id: str,
        reason: str,
    ) -> dict:
        """
        Validate return eligibility and record a return request in DynamoDB.

        Args:
            customer_id: Identifier of the customer requesting the return.
            order_id: Identifier of the order to return.
            reason: Customer-provided reason for the return.

        Returns:
            A dictionary describing success or failure. On success, it
            includes the return reference, status, and order/customer IDs.
        """
        if not customer_id or not order_id or not reason.strip():
            return {
                "success": False,
                "message": "Customer ID, order ID, and return reason are required.",
            }

        orders_table = dynamodb.Table(config.ORDERS_TABLE)
        customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

        # Retrieve the requested order.
        order_response = orders_table.get_item(
            Key={"order_id": order_id}
        )
        order = order_response.get("Item")

        if not order:
            return {
                "success": False,
                "message": f"Order {order_id} was not found.",
            }

        # Confirm that this order belongs to the requesting customer.
        if str(order.get("customer_id", "")) != str(customer_id):
            return {
                "success": False,
                "message": "The order does not belong to the supplied customer.",
            }

        # Retrieve the customer's tier.
        customer_response = customers_table.get_item(
            Key={"customer_id": customer_id}
        )
        customer = customer_response.get("Item")

        if not customer:
            return {
                "success": False,
                "message": f"Customer {customer_id} was not found.",
            }

        tier = str(customer.get("tier", "")).strip().lower()

        if tier == "premium":
            return_window_days = 60
        elif tier == "standard":
            return_window_days = 30
        else:
            return {
                "success": False,
                "message": (
                    "The customer's tier is missing or unrecognized. "
                    "Return eligibility could not be verified."
                ),
            }

        # Respect an explicit ineligibility flag in the order record.
        return_eligible = str(
            order.get("return_eligible", "")
        ).strip().lower()

        if return_eligible == "false":
            return {
                "success": False,
                "message": "The order is marked as ineligible for return.",
            }

        if return_eligible != "true":
            return {
                "success": False,
                "message": (
                    "The order's return-eligibility flag is missing or "
                    "unrecognized. Eligibility could not be verified."
                ),
            }

        # Validate the order date against the customer's tier window.
        order_date_text = str(order.get("order_date", ""))[:10]

        try:
            order_date = datetime.strptime(
                order_date_text,
                "%Y-%m-%d",
            ).date()
        except (TypeError, ValueError):
            return {
                "success": False,
                "message": (
                    "The order date is missing or invalid. "
                    "Return eligibility could not be verified."
                ),
            }

        today = datetime.now(timezone.utc).date()
        days_since_order = (today - order_date).days

        if days_since_order < 0:
            return {
                "success": False,
                "message": "The order date is in the future.",
            }

        if days_since_order > return_window_days:
            return {
                "success": False,
                "message": (
                    f"The order is outside the {return_window_days}-day "
                    f"return window for {tier.title()} customers."
                ),
            }

        # Avoid creating another request for an order with an active request.
        existing_return_status = str(
            order.get("return_status", "")
        ).strip().upper()

        if existing_return_status in {
            "RETURN_REQUESTED",
            "RETURN_APPROVED",
            "REFUNDED",
        }:
            return {
                "success": False,
                "message": (
                    f"This order already has return status "
                    f"{existing_return_status}."
                ),
                "return_status": existing_return_status,
            }

        return_reference = f"RET-{uuid.uuid4().hex[:12].upper()}"
        requested_at = datetime.now(timezone.utc).isoformat()

        # Preserve the original shipping/order status.
        orders_table.update_item(
            Key={"order_id": order_id},
            UpdateExpression=(
                "SET return_status = :return_status, "
                "return_reason = :reason, "
                "return_reference = :reference, "
                "return_requested_at = :requested_at"
            ),
            ExpressionAttributeValues={
                ":return_status": "RETURN_REQUESTED",
                ":reason": reason.strip(),
                ":reference": return_reference,
                ":requested_at": requested_at,
            },
        )

        return {
            "success": True,
            "return_reference": return_reference,
            "return_status": "RETURN_REQUESTED",
            "order_id": order_id,
            "customer_id": customer_id,
            "message": (
                "The return request has been recorded. "
                "This confirms a return request, not a completed refund."
            ),
            "instructions": (
                "Keep the return reference for future correspondence. "
                "Further return instructions should follow NovaMart's "
                "applicable return policy."
            ),
        }

    return Agent(
        model=refund_model,
        tools=[get_inventory_context, initiate_refund],
        system_prompt=refund_system_prompt,
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────


def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    parallel, each querying its own Knowledge Base.
    """

    # Returns policy retrieval tool
    @tool
        # Returns policy retrieval tool
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: Natural-language question about returns policy.

        Returns:
            A formatted string containing the retrieved policy passages.
        """
        with trace_kb_retrieval(config.RETURNS_KB_ID):
            results = retrieve_from_knowledge_base(
                config.RETURNS_KB_ID,
                query,
                top_k=3,
            )
        return format_kb_results(results)

    returns_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
            **_apply_guardrail(),
        ),
        tools=[retrieve_returns_policy],
        system_prompt=(
            "You are a Returns Policy Retriever. "
            "Use retrieve_returns_policy to retrieve relevant policy passages. "
            "Report only information supported by the retrieved passages. "
            "If no relevant information is found, say so explicitly. "
            "Do not invent policy rules or answer unrelated questions."
        ),
    )

    # Shipping policy retrieval tool
    @tool
        # Shipping policy retrieval tool
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: Natural-language question about shipping policy.

        Returns:
            A formatted string containing the retrieved policy passages.
        """
        with trace_kb_retrieval(config.SHIPPING_KB_ID):
            results = retrieve_from_knowledge_base(
                config.SHIPPING_KB_ID,
                query,
                top_k=3,
            )
        return format_kb_results(results)

    shipping_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
            **_apply_guardrail(),
        ),
        tools=[retrieve_shipping_policy],
        system_prompt=(
            "You are a Shipping Policy Retriever. "
            "Use retrieve_shipping_policy to retrieve relevant policy passages. "
            "Report only information supported by the retrieved passages. "
            "If no relevant information is found, say so explicitly. "
            "Do not invent policy rules or answer unrelated questions."
        ),
    )

    # Warranty policy retrieval tool
    @tool
        # Warranty policy retrieval tool
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: Natural-language question about warranty policy.

        Returns:
            A formatted string containing the retrieved policy passages.
        """
        with trace_kb_retrieval(config.WARRANTY_KB_ID):
            results = retrieve_from_knowledge_base(
                config.WARRANTY_KB_ID,
                query,
                top_k=3,
            )
        return format_kb_results(results)

    warranty_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
            **_apply_guardrail(),
        ),
        tools=[retrieve_warranty_policy],
        system_prompt=(
            "You are a Warranty Policy Retriever. "
            "Use retrieve_warranty_policy to retrieve relevant policy passages. "
            "Report only information supported by the retrieved passages. "
            "If no relevant information is found, say so explicitly. "
            "Do not invent policy rules or answer unrelated questions."
        ),
    )

    # One exposed tool dispatches to all three retriever agents in parallel.
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query the Returns, Shipping, and Warranty knowledge bases in parallel.

        Args:
            query: Natural-language question to search across all three
                policy domains.

        Returns:
            A combined string containing each domain's retrieval result,
            including a failure message if an individual retrieval fails.
        """
        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }

        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            try:
                response = agent(query)
                return domain, str(response)
            except Exception as exc:
                logging.exception(
                    "%s policy retriever failed",
                    domain,
                )
                return domain, f"{domain} retrieval failed: {exc}"

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _run_retriever,
                    domain,
                    agent,
                    query,
                ): domain
                for domain, agent in retrievers.items()
            }

            for future in as_completed(futures):
                domain, result_text = future.result()
                results[domain] = result_text

        trace.kb_done(len(retrievers))

        # Print the completed results sequentially for a readable trace.
        for domain in ["Returns", "Shipping", "Warranty"]:
            trace.kb_result(
                domain,
                results.get(domain, "[No results]"),
            )

        combined_results = "\n\n".join(
            f"=== {domain} Policy ===\n"
            f"{results.get(domain, '[No results returned]')}"
            for domain in ["Returns", "Shipping", "Warranty"]
        )

        return combined_results

    # Coordinator model
    policy_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
        **_apply_guardrail(),
    )

    policy_system_prompt = """
You are the NovaMart Policy Agent.

Your responsibility is to answer customer questions about NovaMart's
returns, shipping, and warranty policies.

Instructions:
1. Use search_all_policies to retrieve relevant evidence before answering.
2. The tool searches the Returns, Shipping, and Warranty policy knowledge
   bases in parallel.
3. Base your answer only on the retrieved policy evidence.
4. Do not invent rules, exceptions, deadlines, or eligibility requirements.
5. If the retrieved documents conflict or leave an important detail unclear,
   explicitly state the uncertainty rather than choosing an unsupported answer.
6. Summarize the relevant policy clearly and concisely.
7. Do not claim that an order has been refunded, shipped, or modified.
   You are responsible for policy information, not order operations.
"""

    # The coordinator exposes exactly one tool.
    return Agent(
        model=policy_model,
        tools=[search_all_policies],
        system_prompt=policy_system_prompt,
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────


def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Reads the full WorkflowState and composes the final customer-facing reply.
    """

    communication_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3,
        **_apply_guardrail(),
    )

    communication_system_prompt = """
You are the NovaMart Communication Agent.

Your responsibility is to compose the final customer-facing response using
the findings already recorded in WorkflowState.

Instructions:
1. Call get_full_workflow_context with the current session_id before drafting
   your response.
2. Review the available inventory, policy, and refund findings.
3. Be clear, polite, empathetic, and concise.
4. Explain what was confirmed and what remains unresolved.
5. Never invent customer details, order information, tracking details,
   policy rules, return references, or refund outcomes.
6. Do not state that a refund has been completed when only a return request
   was recorded.
7. If required information is absent, acknowledge the limitation rather
   than guessing.
8. Your response is the final customer-facing message. Do not delegate
   the composition of that message to another agent.
"""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete workflow context for a customer-support session.

        Args:
            session_id: Identifier of the customer-support session.

        Returns:
            A dictionary containing the session's customer ID and agent
            findings, or an error message if the state is unavailable.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return {
                "success": False,
                "message": "No workflow state was found for this session.",
                "inventory_agent": {},
                "policy_agent": {},
                "refund_agent": {},
                "communication_agent": {},
            }

        return {
            "success": True,
            "session_id": state.get("session_id"),
            "customer_id": state.get("customer_id"),
            "inventory_agent": state.get("inventory_agent", {}),
            "policy_agent": state.get("policy_agent", {}),
            "refund_agent": state.get("refund_agent", {}),
            "communication_agent": state.get("communication_agent", {}),
        }

    return Agent(
        model=communication_model,
        tools=[get_full_workflow_context],
        system_prompt=communication_system_prompt,
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────


def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent,
) -> Agent:
    """
    Build the NovaMart Orchestrator Agent.

    Initializes each session, routes requests to the appropriate worker
    agents, persists their findings, and finishes with the Communication Agent.
    """

    orchestrator_model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0,
        **_apply_guardrail(),
    )

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Initialize WorkflowState for a customer-support session.

        Args:
            session_id: Identifier to assign to the session.
            customer_id: Identifier of the customer associated with it.

        Returns:
            A string confirming initialization or explaining why it failed.
        """
        if not session_id or not customer_id:
            return "Error: session_id and customer_id are required."

        existing_state = _read_workflow_state(session_id)

        if existing_state:
            if str(existing_state.get("customer_id")) != str(customer_id):
                return (
                    "Error: this session is already associated with "
                    "a different customer."
                )

            return (
                f"Session {session_id} is already initialized. "
                "Existing workflow state has been preserved."
            )

        _create_workflow_state(session_id, customer_id)

        return (
            f"Session {session_id} initialized successfully "
            f"for customer {customer_id}."
        )

    @tool
    def route_to_inventory_agent(
        session_id: str,
        customer_id: str,
        request: str,
    ) -> str:
        """
        Route an order or account enquiry to the Inventory Agent.

        Args:
            session_id: Initialized customer-support session identifier.
            customer_id: Customer whose information is being requested.
            request: Original customer request to investigate.

        Returns:
            The Inventory Agent's findings, or an error message if routing
            or workflow-state persistence fails.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return "Error: initialize the session before routing requests."

        if str(state.get("customer_id")) != str(customer_id):
            return "Error: customer does not match the initialized session."

        prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Customer request: {request}\n"
            f"Current WorkflowState snapshot: {state}\n\n"
            "Investigate this request using your available tools. "
            "Report the facts you can verify. Do not invent missing data."
        )

        result = str(inventory_agent(prompt))

        updated_state = _update_workflow_state(
            session_id,
            {"inventory_agent": result},
            expected_version=state["version"],
        )

        if (
            not updated_state
            or int(updated_state.get("version", 0)) <= int(state["version"])
        ):
            return (
                "Inventory Agent returned a result, but "
                "WorkflowState version did not advance."
            )

        _update_workflow_state(
            session_id,
            {"inventory_agent": result},
            expected_version=current_state["version"],
        )

        return result

    @tool
    def route_to_policy_agent(
        session_id: str,
        request: str,
    ) -> str:
        """
        Route a policy question to the Policy Agent.

        Args:
            session_id: Initialized customer-support session identifier.
            request: Customer question about returns, shipping, or warranty.

        Returns:
            The Policy Agent's response, or an error message if routing
            or workflow-state persistence fails.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return "Error: initialize the session before routing requests."

        prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {state.get('customer_id')}\n"
            f"Customer request: {request}\n"
            f"Current WorkflowState snapshot: {state}\n\n"
            "Retrieve relevant NovaMart policy evidence and provide "
            "a grounded answer. Clearly identify any uncertainty."
        )

        result = str(policy_agent(prompt))

        updated_state = _update_workflow_state(
            session_id,
            {"policy_agent": result},
            expected_version=state["version"],
        )

        if (
            not updated_state
            or int(updated_state.get("version", 0)) <= int(state["version"])
        ):
            return (
                "Policy Agent returned a result, but "
                "WorkflowState version did not advance."
            )

        _update_workflow_state(
            session_id,
            {"policy_agent": result},
            expected_version=current_state["version"],
        )

        return result

    @tool
    def route_to_refund_agent(
        session_id: str,
        customer_id: str,
        request: str,
    ) -> str:
        """
        Route a return or refund request to the Refund Agent.

        Args:
            session_id: Initialized customer-support session identifier.
            customer_id: Customer requesting the return or refund.
            request: Original customer request to evaluate.

        Returns:
            The Refund Agent's findings, or an error message if routing
            or workflow-state persistence fails.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return "Error: initialize the session before routing requests."

        if str(state.get("customer_id")) != str(customer_id):
            return "Error: customer does not match the initialized session."

        prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Customer request: {request}\n"
            f"Current WorkflowState snapshot: {state}\n\n"
            "First inspect the inventory findings in WorkflowState. "
            "Verify the order and return eligibility before initiating "
            "any return. Do not claim that a refund has been completed "
            "unless that outcome is explicitly confirmed."
        )

        result = str(refund_agent(prompt))

        updated_state = _update_workflow_state(
            session_id,
            {"refund_agent": result},
            expected_version=state["version"],
        )

        if (
            not updated_state
            or int(updated_state.get("version", 0)) <= int(state["version"])
        ):
            return (
                "Refund Agent returned a result, but "
                "WorkflowState version did not advance."
            )

        _update_workflow_state(
            session_id,
            {"refund_agent": result},
            expected_version=current_state["version"],
        )

        return result

    @tool
    def route_to_communication_agent(
        session_id: str,
        customer_id: str,
        original_request: str,
    ) -> str:
        """
        Ask the Communication Agent to compose the final customer response.

        Args:
            session_id: Initialized customer-support session identifier.
            customer_id: Customer who will receive the response.
            original_request: Customer's original question or request.

        Returns:
            The final customer-facing response, or an error message if
            routing or workflow-state persistence fails.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return "Error: initialize the session before routing requests."

        if str(state.get("customer_id")) != str(customer_id):
            return "Error: customer does not match the initialized session."

        prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Original customer request: {original_request}\n"
            f"Current WorkflowState snapshot: {state}\n\n"
            "Read the complete WorkflowState using your tool and compose "
            "the final customer-facing response. Use only verified findings. "
            "Be clear about any unresolved questions or limitations."
        )

        result = str(communication_agent(prompt))

        updated_state = _update_workflow_state(
            session_id,
            {"communication_agent": result},
            expected_version=state["version"],
        )

        if (
            not updated_state
            or int(updated_state.get("version", 0)) <= int(state["version"])
        ):
            return (
                "Communication Agent returned a result, but "
                "WorkflowState version did not advance."
            )

        _update_workflow_state(
            session_id,
            {"communication_agent": result},
            expected_version=current_state["version"],
        )

        return result

    orchestrator_system_prompt = """
You are the NovaMart Orchestrator Agent.

Your job is to coordinate specialist agents. You must not independently
compose the final customer-facing answer.

MANDATORY WORKFLOW:
1. Call initialize_session first for every customer request.
2. For order-status, delivery, tracking, and account-information questions,
   call route_to_inventory_agent.
3. For return or refund requests, call route_to_inventory_agent first,
   then route_to_refund_agent. The Refund Agent must inspect the inventory
   findings before deciding eligibility.
4. For questions about returns policy, shipping policy, warranty, or other
   policy details, call route_to_policy_agent.
5. For account questions, use the Inventory Agent, not the Policy Agent.
6. For simple mathematical questions that require no customer data or policy
   lookup, answer the calculation directly without routing to a worker agent.
7. Always finish by calling route_to_communication_agent for every request,
   including simple mathematical questions.
8. Do not skip session initialization or the final Communication Agent.
9. Do not invent customer, order, or policy facts.
10. Do not tell the customer a refund has been completed when only a return
    request has been recorded.

Use the tool results to decide which specialist agents are needed.
The Communication Agent is responsible for the final customer-facing reply.
"""

    return Agent(
        model=orchestrator_model,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
        system_prompt=orchestrator_system_prompt,
    )

# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════



def create_guardrail() -> tuple[str, str]:
    """Create or update the NovaMart guardrail and publish a numbered version."""
    bedrock_client = boto3.client("bedrock", region_name=config.AWS_REGION)

    content_policy = {
        "filtersConfig": [
            {
                "type": category,
                "inputStrength": strength,
                "outputStrength": strength,
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            }
            for category, strength in [
                ("SEXUAL", "HIGH"),
                ("VIOLENCE", "HIGH"),
                ("HATE", "HIGH"),
                ("INSULTS", "MEDIUM"),
                ("MISCONDUCT", "MEDIUM"),
            ]
        ]
    }

    sensitive_information_policy = {
        "piiEntitiesConfig": [
            {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
            {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
            {"type": "EMAIL", "action": "ANONYMIZE"},
            {"type": "PHONE", "action": "ANONYMIZE"},
        ]
    }

    topic_policy = {
        "tierConfig": {"tierName": "STANDARD"},
        "topicsConfig": [
            {
                "name": "competitor_products",
                "definition": (
                    "Requests for recommendations, comparisons, or product "
                    "information intended to promote competing retailers or "
                    "their products instead of NovaMart."
                ),
                "examples": [
                    "Recommend a competing retailer instead of NovaMart.",
                    "Compare NovaMart products with a competitor's products.",
                ],
                "type": "DENY",
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            },
            {
                "name": "pricing_negotiations",
                "definition": (
                    "Requests to haggle over or change an advertised price, "
                    "or to obtain an unauthorized special price or discount. "
                    "This does not include arithmetic using a specified price, "
                    "calculating a stated discount, or explaining an advertised "
                    "offer without negotiating or changing its terms."
                ),
                "examples": [
                    "Can you negotiate the advertised price down for me?",
                    "Change the listed price to a lower price just for me.",
                    "Give me an additional discount that is not advertised "
                    "or authorized.",
                ],
                "type": "DENY",
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            },
            {
                "name": "legal_threats",
                "definition": (
                    "Requests to draft, formulate, or escalate threats of "
                    "legal action or litigation against NovaMart or its staff."
                ),
                "examples": [
                    "Write a threat saying I will sue NovaMart unless "
                    "the company gives me a refund.",
                    "Draft a message threatening litigation against "
                    "a NovaMart employee.",
                ],
                "type": "DENY",
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            },
        ],
    }

    word_policy = {
        "managedWordListsConfig": [
            {
                "type": "PROFANITY",
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            }
        ]
    }

    guardrail_config = {
        "name": config.GUARDRAIL_NAME,
        "description": "NovaMart customer support safety guardrail.",
        "contentPolicyConfig": content_policy,
        "sensitiveInformationPolicyConfig": sensitive_information_policy,
        "topicPolicyConfig": topic_policy,
        "wordPolicyConfig": word_policy,
        "crossRegionConfig": {
            "guardrailProfileIdentifier": "us.guardrail.v1:0"
        },
        "blockedInputMessaging": (
            "I'm sorry, but I can't help with that request. "
            "Please ask a NovaMart customer-support question."
        ),
        "blockedOutputsMessaging": (
            "I'm sorry, but I can't provide that response. "
            "I can still help with an appropriate NovaMart support request."
        ),
    }

    # Find an existing guardrail with the configured name.
    paginator = bedrock_client.get_paginator("list_guardrails")
    existing_guardrail = None

    for page in paginator.paginate():
        existing_guardrail = next(
            (
                guardrail
                for guardrail in page.get("guardrails", [])
                if guardrail.get("name") == config.GUARDRAIL_NAME
            ),
            None,
        )
        if existing_guardrail:
            break

    if existing_guardrail:
        guardrail_id = existing_guardrail["id"]

        # Update the DRAFT configuration, including the STANDARD topic tier.
        bedrock_client.update_guardrail(
            guardrailIdentifier=guardrail_id,
            **guardrail_config,
        )
        print(f"Updated guardrail draft: {guardrail_id}")
    else:
        response = bedrock_client.create_guardrail(**guardrail_config)
        guardrail_id = response["guardrailId"]
        print(f"Created guardrail: {guardrail_id}")

    # Publish a numbered version; do not return DRAFT.
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="NovaMart Task 3 guardrail version",
    )
    guardrail_version = str(version_response["version"])

    if not guardrail_version.isdigit():
        raise RuntimeError(
            f"Expected a numbered guardrail version, got {guardrail_version!r}"
        )

    print(
        f"Guardrail ready: ID={guardrail_id}, version={guardrail_version}"
    )
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system using the supplied AgentCore CLI helper.

    Returns:
        The AgentCore Runtime ARN.
    """
    if not guardrail_id or not guardrail_version.isdigit():
        raise ValueError(
            "A guardrail ID and numbered guardrail version are required."
        )

    env_vars = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": guardrail_version,
    }

    agentcore_cli.stage_runtime_code()

    agentcore_cli.configure_runtime(
        env_vars=env_vars,
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )

    agentcore_cli.deploy()

    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError(
            "AgentCore CLI deployment completed, but no runtime ARN was found."
        )

    print(f"AgentCore Runtime deployed: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════


def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses summary memory with a seven-day event expiry.

    Returns:
        The memory resource ARN.
    """
    memory_name = config.MEMORY_NAME

    # Reuse an existing memory with the configured name.
    paginator = agentcore_control.get_paginator("list_memories")
    existing_memory = None

    for page in paginator.paginate():
        existing_memory = next(
            (
                memory
                for memory in page.get("memories", [])
                if (
                    memory.get("name") == memory_name
                    or memory.get("id", "").startswith(config.MEMORY_NAME)
                )
            ),
            None,
        )
        if existing_memory:
            break

    if existing_memory:
        memory_id = existing_memory.get("id") or existing_memory.get("memoryId")
        memory_arn = existing_memory.get("arn")
        print(f"AgentCore Memory already exists: {memory_arn}")
    else:
        response = agentcore_control.create_memory(
            name=memory_name,
            description=(
                "NovaMart customer-support conversational memory "
                "with session summaries."
            ),
            eventExpiryDuration=7,
            memoryStrategies=[
                {
                    "summaryMemoryStrategy": {
                        "name": "SessionSummary",
                        "description": (
                            "Summarize customer-support conversations "
                            "to preserve useful context across turns."
                        ),
                        "namespaces": [
                            "/summaries/{actorId}/{sessionId}/"
                        ],
                    }
                }
            ],
            clientToken=uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{config.PROJECT_NAME}:{memory_name}",
            ).hex,
        )

        memory = response.get("memory", response)
        memory_id = memory.get("id") or memory.get("memoryId")
        memory_arn = memory.get("arn")

        if not memory_id or not memory_arn:
            raise RuntimeError(
                "AgentCore Memory creation did not return the expected "
                "memory ID and ARN."
            )

        print(f"Created AgentCore Memory: {memory_arn}")

    # Wait for the memory resource to become ACTIVE.
    timeout_seconds = 300
    poll_interval_seconds = 5
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = agentcore_control.get_memory(memoryId=memory_id)
        memory = response.get("memory", response)
        status = memory.get("status")

        if status == "ACTIVE":
            memory_arn = memory.get("arn", memory_arn)
            print(f"AgentCore Memory is ACTIVE: {memory_arn}")
            return memory_arn

        if status in {"FAILED", "DELETING", "DELETED"}:
            raise RuntimeError(
                f"AgentCore Memory entered unexpected status: {status}"
            )

        print(f"Waiting for AgentCore Memory. Current status: {status}")
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"AgentCore Memory did not become ACTIVE within "
        f"{timeout_seconds} seconds."
    )


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════


def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore observability:
    - CloudWatch logging at INFO level
    - X-Ray tracing with 100% sampling
    """
    from agent_observability import apply_observability_config

    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }

    try:
        summary = apply_observability_config(
            runtime_arn,
            logging_configuration,
    )
    except Exception:
        logging.exception(
            "Failed to configure observability for runtime %s",
            runtime_arn,
        )
        raise

    print(f"CloudWatch logging enabled: {summary['log_group']}")
    print("X-Ray tracing enabled with 100% sampling.")
    print(f"Runtime environment: {summary['runtime_env']}")

# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


def run_serve():
    """Start the AgentCore HTTP runtime and serve incoming requests."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    print("[RUNTIME] Initializing NovaMart agent graph...", flush=True)

    inventory_agent = build_inventory_agent()
    refund_agent = build_refund_agent()
    policy_agent = build_policy_agent()
    communication_agent = build_communication_agent()

    orchestrator = build_orchestrator_agent(
        inventory_agent,
        refund_agent,
        policy_agent,
        communication_agent,
    )

    print("[RUNTIME] Agent graph initialized.", flush=True)

    @app.entrypoint
    def invoke(payload):
        if not isinstance(payload, dict):
            return {"error": "Expected a JSON object containing a prompt."}

        query = (
            payload.get("prompt")
            or payload.get("input")
            or payload.get("message")
        )

        if not isinstance(query, str) or not query.strip():
            return {
                "error": "Provide a non-empty 'prompt', 'input', or 'message'."
            }

        customer_id = str(payload.get("customer_id") or "CUST-001")
        session_id = str(payload.get("session_id") or uuid.uuid4().hex[:8])

        prompt = (
            f"[Session ID: {session_id}] "
            f"[Customer ID: {customer_id}] "
            f"{query.strip()}"
        )

        with tracer.trace_request(
            session_id=session_id,
            customer_id=customer_id,
            request=query.strip(),
        ):
            response = orchestrator(prompt)

            final_state = _read_workflow_state(session_id) or {}
            communication_result = final_state.get("communication_agent", "")
            answer = _strip_xml_tags(
                communication_result or str(response)
            )

            return {
                "response": answer,
                "session_id": session_id,
            }

    app.run()

if __name__ == '__main__':
    runtime_marker = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        agentcore_cli.RUNTIME_MARKER,
    )

    if os.path.isfile(runtime_marker):
        run_serve()

    elif len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)

            final_state = _read_workflow_state(session_id) or {}
            communication_result = final_state.get("communication_agent", "")
            answer = _strip_xml_tags(
                communication_result or str(response)
            )
            print(f"\nResponse:\n{answer}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
