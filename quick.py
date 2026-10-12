
import boto3

client = boto3.client(
    "bedrock-agentcore-control",
    region_name="us-east-1",
)

def show_shape(shape, indent=0, depth=0):
    if shape is None or depth > 4:
        return

    prefix = " " * indent

    if shape.type_name == "structure":
        print(f"{prefix}Required: {shape.required_members}")
        for name, member in shape.members.items():
            print(f"{prefix}{name}: {member.type_name}")
            show_shape(member, indent + 2, depth + 1)

    elif shape.type_name == "list":
        show_shape(shape.member, indent + 2, depth + 1)

    elif shape.type_name == "map":
        show_shape(shape.value, indent + 2, depth + 1)

for operation_name in ["CreateAgentRuntime", "CreateMemory"]:
    print(f"\n{'=' * 50}\n{operation_name}")
    operation = client.meta.service_model.operation_model(operation_name)
    show_shape(operation.input_shape)