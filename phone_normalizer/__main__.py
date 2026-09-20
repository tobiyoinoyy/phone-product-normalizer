"""Allow ``python -m phone_normalizer`` to invoke the delivery CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
