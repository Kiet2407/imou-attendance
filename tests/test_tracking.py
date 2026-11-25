import unittest
from attendance.tracking import EntryTracker


class TrackingTests(unittest.TestCase):
    def sequence(self, xs, identity=('a','An',.8), **kwargs):
        tracker = EntryTracker(**kwargs)
        events = []
        for i, x in enumerate(xs):
            events += tracker.update([((x,.5),identity)], i*.1)
        return events

    def test_entry_once(self):
        self.assertEqual(len(self.sequence([.3,.4,.5,.6,.65,.55,.4,.5,.6])),1)

    def test_exit_not_entry(self):
        self.assertEqual(self.sequence([.7,.6,.5,.4]),[])

    def test_appears_inside(self):
        self.assertEqual(self.sequence([.6,.65,.7,.75]),[])

    def test_unknown(self):
        self.assertEqual(self.sequence([.3,.4,.5,.6],None),[])

    def test_not_enough_observations(self):
        self.assertEqual(self.sequence([.44,.56]),[])

    def test_reverse_direction(self):
        self.assertEqual(len(self.sequence([.7,.6,.5,.4],direction=-1)),1)

    def test_identity_switch(self):
        t=EntryTracker(confirmations=2)
        t.update([((.35,.5),('a','An',.8))],0)
        t.update([((.45,.5),('a','An',.8))],.1)
        self.assertEqual(t.update([((.56,.5),('b','Binh',.8))],.2),[])

    def test_expired_track(self):
        t=EntryTracker(confirmations=1)
        t.update([((.44,.5),('a','An',.8))],0)
        self.assertEqual(t.update([((.56,.5),('a','An',.8))],2),[])

    def test_two_faces_ambiguous(self):
        t=EntryTracker(confirmations=1)
        t.update([((.44,.5),('a','An',.8))],0)
        self.assertEqual(t.update([((.56,.5),('a','An',.8)),((.55,.5),None)],.1),[])

    def test_unknown_cannot_use_stale_identity(self):
        t=EntryTracker(confirmations=3)
        for i in range(3):
            t.update([((.41,.5),('a','An',.8))],i*.1)
        t.update([((.45,.5),None)],.3)
        self.assertEqual(t.update([((.56,.5),None)],.4),[])
