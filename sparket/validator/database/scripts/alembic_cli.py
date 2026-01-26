"""
Alembic CLI wrapper for the validator.

This script is used to run Alembic commands for the validator without having to specify the path to the alembic.ini file.
"""


import sys
from alembic import command
from alembic.config import Config


def main() -> None:
    cfg = Config("sparket/validator/database/alembic.ini")
    # Pass through alembic subcommands/args after the script name
    args = sys.argv[1:]
    if not args:
        print("Usage: sparket-alembic <alembic-args>")
        sys.exit(1)

    # Minimal dispatcher for common commands; for anything else, defer to CLI
    # Examples:
    #   sparket-alembic upgrade head
    #   sparket-alembic downgrade -1
    cmd = args[0]
    remainder = args[1:]
    if cmd == "upgrade" and remainder:
        command.upgrade(cfg, remainder[0])
        return
    if cmd == "downgrade" and remainder:
        command.downgrade(cfg, remainder[0])
        return
    if cmd == "current":
        command.current(cfg)
        return
    if cmd == "history":
        command.history(cfg)
        return
    if cmd == "revision":
        # allow: sparket-alembic revision --autogenerate -m "msg"
        from alembic.config import CommandLine
        CommandLine(prog="sparket-alembic").main(argv=args)
        return

    # Fallback to the full alembic CLI parser for all other cases
    from alembic.config import CommandLine
    CommandLine(prog="sparket-alembic").main(argv=args)


if __name__ == "__main__":
    main()


