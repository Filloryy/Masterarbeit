# Masterarbeit
To-do:
compare performance for graph types > single_node>fullbodygraph
"fix" feature dimension
"fix" pooling
parameter search
optimizing the model

implement challenges


Miscellaneous:
-training with GNN runs twice?
-modulate the code for different experiments
-better evaluate tool

Documentation is wrong?????
Orientation of Ant (in walking direction)
0 : front right hip -> obs 11 (shouldve been: action br hip obs br hip)
1 : front right knee -> obs 12 (action br knee obs br knee)
2 : front left hip -> obs 5 (action fl hip observation fl hip)
3 : front left knee -> obs 6 (action fl hip observation fl hip)
4 : back left hip -> obs 7 (action fr hip observation fr hip)
5 :
6 :
7 :
hips:   negative torque twists clockwise
        positive torque twists ccw
knees:  negative torque flexes hind legs stretches front legs
        positive torque stretches hind legs flexes front legs

SHOULD_BE								
                br_hip	br_knee	fl_hip	fl_knee	fr_hip	fr_knee	bl_hip	bl_knee
Action	        0       1   	2       3   	4   	5   	6   	7
Observation	11  	12  	5   	6   	7   	8   	9   	10
                                
                                
WalkingD	fr_hip	fr_knee	fl_hip	fl_knee	bl_hip	bl_knee	br_hip	br_knee
Action	        0       1       2   	3   	4   	5       6       7
Observation	11      12      5       6   	7   	8       9       10

does this even matter?
-knees and hips never seperated
-obs and action were coupled
-connections were not changed since they always go over the torso node


