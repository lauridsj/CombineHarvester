from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

class InterferencePlusFixed(PhysicsModel):
    def __init__(self):
        pass

    def doParametersOfInterest(self):
        '''Create POI and other parameters, and define the POI set.'''
        # --- coupling modifier + signal strength as POIs --- 
        self.modelBuilder.doVar('g[1,0,5]')
        self.modelBuilder.doVar('r[1,0,10]')
        self.modelBuilder.doSet('POI', 'r,g')
        self.modelBuilder.factory_('expr::g_sq("(@0*@0)*@1", g, r)')
        self.modelBuilder.factory_('expr::g_neg_sq("(-@0*@0)*@1", g, r)')
        self.modelBuilder.factory_('expr::g_neg_tothe4("(-@0*@0*@0*@0)*@1", g, r)')
        self.modelBuilder.factory_('expr::g_tothe4("(@0*@0*@0*@0)*@1", g, r)')

    def getYieldScale(self, bin, process):
        "Return the name of a RooAbsReal to scale the yield"
        if not self.DC.isSignal[process]:
            return 1
        if '_neg' in process:
            if '_res' in process:
                print 'Scaling', process, 'with negative coupling modifier to the 4'
                print 'WARNING, negative signal part, sure this is what you want?'
                return 'g_neg_tothe4'
            else:
                print 'Scaling', process, 'with negative coupling modifier squared'
                return 'g_neg_sq'
        elif '_res' in process:
            print 'Scaling', process, 'with coupling modifier to the 4'
            return 'g_tothe4'
        print 'Scaling', process, 'with coupling modifier squared'
        return 'g_sq'

interferencePlusFixed = InterferencePlusFixed()
