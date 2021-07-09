from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

class InterferenceModel(PhysicsModel):
    def __init__(self):
        pass

    def doParametersOfInterest(self):
        '''Create POI and other parameters, and define the POI set.'''
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar('r[1,0,20]')
        self.modelBuilder.doSet('POI', 'r')
        self.modelBuilder.factory_('expr::r_sq("(@0*@0)", r)')
        self.modelBuilder.factory_('expr::r_neg_sq("(-@0*@0)", r)')
        self.modelBuilder.factory_('expr::r_neg_tothe4("(-@0*@0*@0*@0)", r)')
        self.modelBuilder.factory_('expr::r_tothe4("(@0*@0*@0*@0)", r)')
        
    def getYieldScale(self, bin, process):
        "Return the name of a RooAbsReal to scale the yield"
        if not self.DC.isSignal[process]:
            return 1
        if '_neg' in process:
            if '-sgn' in process:
                print 'Scaling', process, 'with negative signal strength to the 4'
                print 'WARNING, negative signal part, sure this is what you want?'
                return 'r_neg_tothe4'
            else:
                print 'Scaling', process, 'with negative signal strength squared'
                return 'r_neg_sq'
        elif '-sgn' in process:
            print 'Scaling', process, 'with signal strength to the 4'
            return 'r_tothe4'
        print 'Scaling', process, 'with signal strength squared'
        return 'r_sq'

interferenceModel = InterferenceModel()
