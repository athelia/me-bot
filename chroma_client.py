import logging

from chromadb.config import Settings


def suppress_chroma_telemetry_noise() -> None:
    """Silence broken Chroma/posthog telemetry errors on stderr."""
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb").setLevel(logging.ERROR)


def chroma_client_settings() -> Settings:
    return Settings(anonymized_telemetry=False)


suppress_chroma_telemetry_noise()
