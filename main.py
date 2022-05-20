ServoMotorNum = 4

class SuperServos():
    def __init__(self, nums) -> None:
        self.ServoAngles = []
        self.ServoExperResultsP = []
        self.ServoExperResultsN = []
        for i in range(nums):
            self.ServoAngles.append(45.0)
            self.ServoExperResultsP.append(0.0)
            self.ServoExperResultsN.append(0.0)
    
    def Move(self, id, action):
        if self.ServoAngles[id] + action > 0 and self.ServoAngles[id] + action < 180:
            self.ServoAngles[id] += action
            return True
        else:
            return False