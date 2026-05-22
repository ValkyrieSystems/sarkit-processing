import re

NEGATIVE_NUMBER_EXPONENTAL_MATCHER = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][+\-]?\d+)?$")


def allow_floating_point_arguments(parser):
    parser._negative_number_matcher = NEGATIVE_NUMBER_EXPONENTAL_MATCHER
