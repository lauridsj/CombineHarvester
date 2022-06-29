from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModelBase_NiceSubclasses

class MultiInterferencePlusFixed(PhysicsModelBase_NiceSubclasses):
    def __init__(self):
        self.signal_parts = ['_neg', '_pos', '_res']
        self.signals = []
        self.pois    = []
        self.verbose = False
        self.nor = False
        super(MultiInterferencePlusFixed, self).__init__()

    def setPhysicsOptions(self, physOptions):
        if len([po for po in physOptions if po.startswith("signal=")]) != 1:
            raise RuntimeError, 'Model expects a signal=s1,s2,...,sN option, provided exactly once'

        for po in physOptions[:]:
            if po.startswith("signal="):
                signals = po.split('=')[1].split(',')

                for ss in signals:
                    if not (ss.startswith("A_m") or ss.startswith("H_m")):
                        raise RuntimeError, 'Model expects only A/H signals'
                    if any([sp in ss for sp in self.signal_parts]):
                        raise RuntimeError, 'Model expects all parts of A/H signal together i.e. it does not handle them piecewise. specify as e.g. signal=A_m400_w5p0,...'

        super(MultiInterferencePlusFixed, self).setPhysicsOptions(physOptions)

    def processPhysicsOptions(self, physOptions):
        processed = []
        physOptions.sort(key = lambda x: x.startswith("no-r"), reverse = True)
        physOptions.sort(key = lambda x: x.startswith("verbose"), reverse = True)
        for po in physOptions:
            if po == "verbose":
                self.verbose = True
                processed.append(po)
            if po == "no-r":
                self.nor = True
                processed.append(po)
            if po.startswith("signal="):
                signals = po.split('=')[1].split(',')

                for ss in signals:
                    self.add_poi_per_signal(ss)

                processed.append(po)

        return processed + super(MultiInterferencePlusFixed, self).processPhysicsOptions(physOptions)

    def add_poi_per_signal(self, signal):
        self.signals.append(signal)
        self.pois.append('g_{ss}'.format(ss = signal))

        if not self.nor:
            self.pois.append('r_{ss}'.format(ss = signal))

    def doParametersOfInterest(self):
        for signal in self.signals:
            self.modelBuilder.doVar('g_{ss}[1,0,5]'.format(ss = signal))

            if self.nor:
                self.modelBuilder.factory_('expr::g2_{ss}("@0*@0", g_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::mg2_{ss}("-@0*@0", g_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::g4_{ss}("@0*@0*@0*@0", g_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::mg4_{ss}("-@0*@0*@0*@0", g_{ss})'.format(ss = signal))
            else:
                self.modelBuilder.doVar('r_{ss}[1,0,10]'.format(ss = signal))
                self.modelBuilder.factory_('expr::g2_{ss}("(@0*@0)*@1", g_{ss}, r_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::mg2_{ss}("(-@0*@0)*@1", g_{ss}, r_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::g4_{ss}("(@0*@0*@0*@0)*@1", g_{ss}, r_{ss})'.format(ss = signal))
                self.modelBuilder.factory_('expr::mg4_{ss}("(-@0*@0*@0*@0)*@1", g_{ss}, r_{ss})'.format(ss = signal))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

    def getPOIList(self):
        return self.pois

    def getYieldScale(self, bin, process):
        if not self.DC.isSignal[process]:
            return 1

        base = process
        for sp in self.signal_parts:
            base = base.replace(sp, "")

        if self.verbose and self.nor:
            print "INFO: using model version without the fixed r term."

        if '_neg' in process:
            if '_res' in process:
                if self.verbose:
                    print 'Scaling', process, 'with negative coupling modifier to the 4'
                    print 'WARNING: negative resonance, are you sure this is what you want?'
                return 'mg4_' + base
            else:
                if self.verbose:
                    print 'Scaling', process, 'with negative coupling modifier squared'
                return 'mg2_' + base
        elif '_res' in process:
            if self.verbose:
                print 'Scaling', process, 'with coupling modifier to the 4'
            return 'g4_' + base

        if self.verbose:
            print 'Scaling', process, 'with coupling modifier squared'
        return 'g2_' + base

multiInterferencePlusFixed = MultiInterferencePlusFixed()
