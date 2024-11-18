from add_edge_attr import add_edge_attr
from add_face_attr import add_face_attr
from add_cross_at_boundaries add_cross_at_boundaries


def pre_processing(self):
    try:
        add_edge_attr()
        if self.activate_Debug_Comments == True:
            print('added edge attr: tensor with boolen values if edge == boundary edge')
    except:
        print('could not add edge attribute')
    try:
        add_face_attr()
        if self.activate_Debug_Comments == True:
            print(
                'added face attr: tensor with boolen values if face == boundary face')
    except:
        print('could not add face attribute')
    try:
        add_cross_at_boundaries()
        if self.activate_Debug_Comments == True:
            print(
                'added face attr: tensor with boolen values if face == boundary face')
    except:

        print('could not add crosses at the boundary')
