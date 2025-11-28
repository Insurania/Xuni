# A dictionary of all the supported possible behavior 
METHODS = {0:'Seek',
           1:'Flee',
           2:'Arrival',
           3:'Departure',
           4:'Wander',
           5:'Avoid',
           6:'Seperation',
           7:'Alignment',
           8:'Cohension',
           9:'Flocking',
           10:'Leader',
           }

Mass = 1.0
Inertia = 1.0
MaxVelocity = 0.01
MaxForce = 0.03
MaxTorque = 30
MaxAngVel = 30.0

Kv1 = 4.0
Kv0 = 4.0
Kp1 = 1.0
KArrival = 0.8
KFlee = 10
KSeek = 10
KDeparture = 15.0
KNoise = 1.0
KWander = 0.05
KAvoid = 0.1
ObstacleRadius = 0.04
RNeighborhood = 1.0
KSeparation = 5.0
KAlignment = 3.0
KCohesion = 3.0
KLeaderSeperation = 1.25
KLeaderArrival = 1.55