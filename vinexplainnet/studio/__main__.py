from __future__ import annotations

import argh

from vinexplainnet.studio import clapperboard


def main() -> None:
    parser = argh.ArghParser()
    parser.add_commands(
        [
            clapperboard.train,
            clapperboard.evaluate,
            clapperboard.explain,
            clapperboard.optimize,
            clapperboard.export,
        ]
    )
    parser.dispatch()


if __name__ == "__main__":
    main()
