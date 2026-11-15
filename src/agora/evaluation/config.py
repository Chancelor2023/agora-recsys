"""Read evaluation cutoffs from the shared standard-library TOML configuration."""
import tomllib


def load_cutoffs(path, scope="primary"):
    if scope not in ("primary", "supplementary"):
        raise ValueError("scope must be primary or supplementary")
    with open(path, "rb") as stream:
        config = tomllib.load(stream)
    settings = config.get("evaluation", {})
    if settings.get("zero_positive_policy") != "exclude_and_report_count":
        raise ValueError("unsupported zero_positive_policy")
    prefix = "supplementary_" if scope == "supplementary" else ""
    values = []
    for metric in ("ndcg_k", "recall_k"):
        key = prefix + metric
        value = settings.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
        values.append(value)
    return tuple(values)
