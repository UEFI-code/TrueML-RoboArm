ServoMotorNum = 4

class SuperServos():
    def __init__(self, nums, delta) -> None:
        self.ServoNum = nums
        self.delta = delta
        self.ServoAngles = []
        self.ServoExperResultsP = []
        self.ServoExperResultsN = []
        for i in range(nums):
            self.ServoExperResultsP.append(0.0)
            self.ServoExperResultsN.append(0.0)
            self.ServoAngles.append(45.0)
    
    def Move(self, id, action):
        if self.ServoAngles[id] + action > 0 and self.ServoAngles[id] + action < 180:
            self.ServoAngles[id] += action
            return True
        else:
            return False
    
    def Measure(self):
        return [1, 2, 3]

    def ComputeEffect(self, Before, After):
        s = 0
        for i, j in zip(Before, After):
            s += (i - j) * (i - j)
        return s

    def Experiment(self):
        for id in range(self.ServoNum):
            self.ServoExperResultsP[id] = 0.0
            self.ServoExperResultsN[id] = 0.0
            statusBefore = self.Measure()
            res = self.Move(id, self.delta)
            if(res):
                statusAfter = self.Measure()
                self.ServoExperResultsP[id] = self.ComputeEffect(statusBefore, statusAfter)
                self.Move(id, 0 - self.delta)
                statusBefore = self.Measure()
            res = self.Move(id, 0 - self.delta)
            if(res):
                statusAfter = self.Measure()
                self.ServoExperResultsN[id] = self.ComputeEffect(statusBefore, statusAfter)
                self.Move(id, self.delta)
    
    def FindBestOperation(self):
        BestIdxP = 0
        BestIdxN = 0
        for i in self.ServoExperResultsP:
            if(self.ServoExperResultsP[i] < self.ServoExperResultsP[BestIdxP]):
                BestIdxP = i
        for j in self.ServoExperResultsN:
            if(self.ServoExperResultsN[j] < self.ServoExperResultsP[BestIdxN]):
                BestIdxN = j
        if self.ServoExperResultsP[BestIdxP] < self.ServoExperResultsN[BestIdxN]:
            return 'P', BestIdxP
        else:
            return 'N', BestIdxN
        

