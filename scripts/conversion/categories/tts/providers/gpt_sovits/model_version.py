T2S_FAMILY_BY_VERSION = {
    "v1": 1,
    "v2": 2,
    "v2Pro": 3,
    "v2ProPlus": 3,
    "v3": 3,
    "v4": 3,
}

def normalize_model_version(version: str) -> str:
    if version not in T2S_FAMILY_BY_VERSION:
        supported = ", ".join(T2S_FAMILY_BY_VERSION)
        raise ValueError(f"Unsupported GPT-SoVITS version '{version}'. Expected one of: {supported}")
    return version


def t2s_family_for_version(version: str) -> int:
    return T2S_FAMILY_BY_VERSION[normalize_model_version(version)]
