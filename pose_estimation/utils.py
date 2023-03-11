import math


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def midpoint(left_body_part, right_body_part):
        midpoint_x = (left_body_part[0] + right_body_part[0]) / 2
        midpoint_y = (left_body_part[1] + right_body_part[1]) / 2
        midpoint_z = (left_body_part[2] + right_body_part[2]) / 2

        return [midpoint_x, midpoint_y, midpoint_z]