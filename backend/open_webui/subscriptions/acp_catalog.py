from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Extract a field from either a dict or an object instance."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def flatten_option_choices(option: dict[str, Any] | Any) -> list[dict[str, Any]]:
    """Flatten selectable choices from an ACP SessionConfigOptionSelect.

    Handles both flat option lists:
        options: [{"value": "m1", "name": "Model 1"}, ...]
    and nested group structures:
        options: [
            {"group": "g1", "name": "Group 1", "options": [{"value": "m1", "name": "Model 1"}]},
            ...
        ]

    Choice values are extracted using the official ACP schema field 'value',
    with backward-compatible fallback to 'id' if 'value' is absent.
    """
    if not option:
        return []

    raw_options = _get_field(option, "options")
    if not isinstance(raw_options, (list, tuple)):
        return []

    flattened: list[dict[str, Any]] = []

    for entry in raw_options:
        if not entry:
            continue

        # Check if entry is a SessionConfigSelectGroup (has nested 'options')
        nested_options = _get_field(entry, "options")
        if isinstance(nested_options, (list, tuple)):
            for child in nested_options:
                if not child:
                    continue
                val = _get_field(child, "value")
                if val is None:
                    val = _get_field(child, "id")
                if val is None:
                    continue
                name = _get_field(child, "name") or str(val)
                desc = _get_field(child, "description")
                choice_dict: dict[str, Any] = {
                    "value": str(val),
                    "name": str(name),
                }
                if desc is not None:
                    choice_dict["description"] = str(desc)
                flattened.append(choice_dict)
        else:
            # Single SessionConfigSelectOption
            val = _get_field(entry, "value")
            if val is None:
                val = _get_field(entry, "id")
            if val is None:
                continue
            name = _get_field(entry, "name") or str(val)
            desc = _get_field(entry, "description")
            choice_dict = {
                "value": str(val),
                "name": str(name),
            }
            if desc is not None:
                choice_dict["description"] = str(desc)
            flattened.append(choice_dict)

    return flattened


def extract_models(
    config_options: list[dict[str, Any]] | Any,
    current_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract advertised model definitions from ACP configOptions.

    Locates the model picker option (category='model' or id='model') and flattens
    its choices, extracting choice.value and choice.name.

    Derives optional reasoning_efforts: list[{"value": str, "name": str}] ONLY for
    the CURRENT model whose thinking options were actually advertised in configOptions.
    Does not copy one model's choices onto other models.

    Returns a list of dicts: [{"id": model_value, "name": model_name, ...}, ...].
    Strictly returns an empty list if no models are advertised; never injects
    hardcoded fallback models.
    """
    if not config_options or not isinstance(config_options, (list, tuple)):
        return []

    # Find the model select option: prefer category="model", fallback to id="model"
    model_option = None
    for opt in config_options:
        if not opt:
            continue
        category = _get_field(opt, "category")
        opt_id = _get_field(opt, "id")
        if category == "model" or opt_id == "model":
            model_option = opt
            break

    if not model_option:
        return []

    choices = flatten_option_choices(model_option)
    if not choices:
        return []

    # Determine current model id if not explicitly passed
    resolved_current_model = current_model_id
    if not resolved_current_model:
        curr_val = _get_field(model_option, "currentValue")
        if curr_val is None:
            curr_val = _get_field(model_option, "current_value")
        if curr_val is not None and isinstance(curr_val, str) and curr_val.strip():
            resolved_current_model = curr_val.strip()
        elif len(choices) == 1:
            only_id = choices[0].get("value") or choices[0].get("id")
            if only_id:
                resolved_current_model = str(only_id).strip()

    # Extract thinking level choices for the current model if advertised
    reasoning_efforts: list[dict[str, str]] = []
    thinking_opt = find_thinking_option(config_options)
    if thinking_opt:
        raw_thinking_choices = flatten_option_choices(thinking_opt)
        seen_efforts: set[str] = set()
        for tc in raw_thinking_choices:
            val = tc.get("value")
            if val is None:
                val = tc.get("id")
            if not isinstance(val, str) or not val.strip():
                continue
            val_clean = val.strip()
            # Enforce bounded string and reject control characters
            if len(val_clean) > 64 or any(ord(ch) < 32 or ord(ch) == 127 for ch in val_clean):
                continue
            if val_clean in seen_efforts:
                continue
            seen_efforts.add(val_clean)
            name = tc.get("name") or val_clean
            if isinstance(name, str) and not any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
                name_clean = name.strip()[:64]
            else:
                name_clean = val_clean
            reasoning_efforts.append({"value": val_clean, "name": name_clean})

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for choice in choices:
        mid = choice.get("value") or choice.get("id")
        if not mid or not isinstance(mid, str) or not mid.strip():
            continue
        mid_clean = mid.strip()
        if len(mid_clean) > 256 or any(ord(ch) < 32 or ord(ch) == 127 for ch in mid_clean):
            continue
        if mid_clean in seen_ids:
            continue
        seen_ids.add(mid_clean)
        name = choice.get("name") or mid_clean
        if isinstance(name, str) and not any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            name_clean = name.strip()[:256]
        else:
            name_clean = mid_clean

        model_entry: dict[str, Any] = {"id": mid_clean, "name": name_clean}
        # Only attach choices for the CURRENT model whose thinking options were actually advertised
        if reasoning_efforts and resolved_current_model and mid_clean == resolved_current_model:
            model_entry["reasoning_efforts"] = [dict(e) for e in reasoning_efforts]

        models.append(model_entry)

    return models


def find_config_option(
    config_options: list[dict[str, Any]] | Any,
    *,
    category: str | None = None,
    option_id: str | None = None,
) -> dict[str, Any] | None:
    """Find a configuration option by category or option id."""
    if not config_options or not isinstance(config_options, (list, tuple)):
        return None

    # First attempt exact category match
    if category is not None:
        for opt in config_options:
            if _get_field(opt, "category") == category:
                return opt if isinstance(opt, dict) else opt.__dict__

    # Next attempt option id match
    if option_id is not None:
        for opt in config_options:
            if _get_field(opt, "id") == option_id:
                return opt if isinstance(opt, dict) else opt.__dict__

    return None


def find_thinking_option(config_options: list[dict[str, Any]] | Any) -> dict[str, Any] | None:
    """Find the thinking level configuration option in ACP configOptions.

    The official Antigravity ACP runtime defines:
        id: "thinking_level"
        category: "thought_level"
    """
    if not config_options or not isinstance(config_options, (list, tuple)):
        return None

    # 1. Primary lookup by category="thought_level"
    for opt in config_options:
        if _get_field(opt, "category") == "thought_level":
            return opt if isinstance(opt, dict) else opt.__dict__

    # 2. Lookup by id="thinking_level"
    for opt in config_options:
        if _get_field(opt, "id") == "thinking_level":
            return opt if isinstance(opt, dict) else opt.__dict__

    # 3. Fallback lookup by id="thought_level"
    for opt in config_options:
        if _get_field(opt, "id") == "thought_level":
            return opt if isinstance(opt, dict) else opt.__dict__

    return None
