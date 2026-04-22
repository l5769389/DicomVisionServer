from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from app.main import fastapi_app
from app.schemas import dicom as dicom_schemas
from app.schemas import view as view_schemas


HEADER = """// Generated from DicomVisionServer FastAPI OpenAPI schemas.
// Do not edit by hand. Regenerate with:
//   uv run python scripts/generate_openapi_types.py --output ../DicomVisionClient/src/shared/generated/backendApi.ts

"""


def _pascal_case(value: str) -> str:
    parts = [part for part in value.replace("-", "_").split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Anonymous"


def _literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


class TypeScriptSchemaWriter:
    def __init__(self, schemas: dict[str, Any]) -> None:
        self.schemas = schemas
        self.inline_counter = 0

    def ref_name(self, ref: str) -> str:
        return ref.rsplit("/", 1)[-1]

    def type_for_schema(self, schema: dict[str, Any] | None, *, required: bool = True) -> str:
        if not schema:
            return "unknown"

        if "$ref" in schema:
            return self.ref_name(str(schema["$ref"]))

        if "const" in schema:
            return _literal(schema["const"])

        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            return " | ".join(_literal(item) for item in enum_values) or "never"

        for union_key in ("anyOf", "oneOf"):
            union_items = schema.get(union_key)
            if isinstance(union_items, list):
                types = [self.type_for_schema(item) for item in union_items]
                return self._dedupe_union(types)

        all_of = schema.get("allOf")
        if isinstance(all_of, list):
            types = [self.type_for_schema(item) for item in all_of]
            return " & ".join(types) if types else "unknown"

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return self._dedupe_union([self.type_for_schema({"type": item}) for item in schema_type])

        if schema_type == "null":
            return "null"
        if schema_type == "boolean":
            return "boolean"
        if schema_type in {"integer", "number"}:
            return "number"
        if schema_type == "string":
            return "string"
        if schema_type == "array":
            prefix_items = schema.get("prefixItems")
            if isinstance(prefix_items, list) and prefix_items:
                return "[" + ", ".join(self.type_for_schema(item) for item in prefix_items) + "]"
            item_type = self.type_for_schema(schema.get("items"))
            if "|" in item_type and not item_type.startswith("(") and not item_type.startswith("{"):
                item_type = f"({item_type})"
            return f"{item_type}[]"
        if schema_type == "object" or "properties" in schema:
            return self.inline_object_type(schema, required=required)

        return "unknown"

    def inline_object_type(self, schema: dict[str, Any], *, required: bool = True) -> str:
        properties = schema.get("properties")
        additional = schema.get("additionalProperties")
        if not isinstance(properties, dict):
            if isinstance(additional, dict):
                return f"Record<string, {self.type_for_schema(additional)}>"
            if additional is True:
                return "Record<string, unknown>"
            return "Record<string, unknown>"

        required_fields = set(schema.get("required") or [])
        parts: list[str] = []
        for name, prop_schema in properties.items():
            optional = "?" if name not in required_fields else ""
            parts.append(f"{self.property_name(name)}{optional}: {self.type_for_schema(prop_schema)}")
        if isinstance(additional, dict):
            parts.append(f"[key: string]: {self.type_for_schema(additional)}")
        elif additional is True:
            parts.append("[key: string]: unknown")
        return "{ " + "; ".join(parts) + " }"

    def render_schema(self, name: str, schema: dict[str, Any]) -> str:
        schema_type = schema.get("type")
        if schema_type == "object" or "properties" in schema:
            return self.render_interface(name, schema)
        return f"export type {name} = {self.type_for_schema(schema)}\n"

    def render_interface(self, name: str, schema: dict[str, Any]) -> str:
        properties = schema.get("properties") or {}
        required_fields = set(schema.get("required") or [])
        lines = [f"export interface {name} {{"]
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                optional = "?" if prop_name not in required_fields else ""
                lines.append(f"  {self.property_name(prop_name)}{optional}: {self.type_for_schema(prop_schema)}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            lines.append(f"  [key: string]: {self.type_for_schema(additional)}")
        elif additional is True:
            lines.append("  [key: string]: unknown")
        lines.append("}\n")
        return "\n".join(lines)

    @staticmethod
    def property_name(name: str) -> str:
        if name.replace("_", "").replace("-", "").isalnum() and "-" not in name:
            return name
        return repr(name)

    @staticmethod
    def _dedupe_union(types: list[str]) -> str:
        deduped: list[str] = []
        for item in types:
            if item not in deduped:
                deduped.append(item)
        return " | ".join(deduped) or "never"


def _operation_name(method: str, path: str, operation: dict[str, Any]) -> str:
    explicit = operation.get("operationId")
    if explicit:
        return _pascal_case(str(explicit))
    path_name = path.strip("/").replace("{", "").replace("}", "").replace("/", "_")
    return _pascal_case(f"{method}_{path_name}")


def _schema_ref_from_media(media: dict[str, Any]) -> dict[str, Any] | None:
    json_media = media.get("application/json")
    if isinstance(json_media, dict):
        schema = json_media.get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def render_types() -> str:
    openapi = fastapi_app.openapi()
    schemas = openapi.get("components", {}).get("schemas", {})
    schemas = dict(schemas)
    for module in (dicom_schemas, view_schemas):
        for value in vars(module).values():
            if not isinstance(value, type) or not issubclass(value, BaseModel) or value is BaseModel:
                continue
            schema = value.model_json_schema(by_alias=True, ref_template="#/components/schemas/{model}")
            defs = schema.pop("$defs", {})
            if isinstance(defs, dict):
                schemas.update(defs)
            schemas.setdefault(value.__name__, schema)
    writer = TypeScriptSchemaWriter(schemas)
    chunks = [HEADER]
    for name in sorted(schemas):
        chunks.append(writer.render_schema(name, schemas[name]))

    chunks.append("export interface ApiOperations {")
    paths = openapi.get("paths", {})
    for path in sorted(paths):
        path_item = paths[path]
        if not isinstance(path_item, dict):
            continue
        for method in sorted(path_item):
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            operation = path_item[method]
            if not isinstance(operation, dict):
                continue
            request_schema = None
            request_body = operation.get("requestBody")
            if isinstance(request_body, dict):
                content = request_body.get("content")
                if isinstance(content, dict):
                    request_schema = _schema_ref_from_media(content)

            response_schema = None
            responses = operation.get("responses")
            if isinstance(responses, dict):
                response = responses.get("200") or responses.get("201") or responses.get("204")
                if isinstance(response, dict):
                    content = response.get("content")
                    if isinstance(content, dict):
                        response_schema = _schema_ref_from_media(content)

            operation_name = _operation_name(method, path, operation)
            request_type = writer.type_for_schema(request_schema) if request_schema else "never"
            response_type = writer.type_for_schema(response_schema) if response_schema else "unknown"
            chunks.append(f"  {operation_name}: {{ method: {repr(method.upper())}; path: {repr(path)}; request: {request_type}; response: {response_type} }}")
    chunks.append("}\n")
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to the generated TypeScript file.")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_types(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
