"""Command-line interface: khmer-id-reader photo.jpg"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="khmer-id-reader",
        description="Read a Cambodian national ID card and print JSON.",
    )
    parser.add_argument("image", help="Path to the card photo (JPG/PNG)")
    parser.add_argument(
        "--corners", type=str, default=None,
        help="Four corner points: x1,y1 x2,y2 x3,y3 x4,y4",
    )
    parser.add_argument(
        "--no-lm", action="store_true",
        help="Disable language model (faster, less accurate)",
    )
    args = parser.parse_args()

    from .api import read_card

    corners = None
    if args.corners:
        pairs = args.corners.strip().split()
        corners = [tuple(map(float, p.split(","))) for p in pairs]

    result = read_card(
        image_path=args.image,
        corners=corners,
        use_lm=not args.no_lm,
    )

    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
