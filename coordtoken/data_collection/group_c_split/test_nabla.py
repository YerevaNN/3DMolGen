from collections import Counter
import sqlite3
import unittest
from run_nabla import select_validation


class SelectionTests(unittest.TestCase):
    def build(self,rows):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER,owner INTEGER,other_train INTEGER,action TEXT)')
        db.executemany('INSERT INTO counts VALUES (?,?,?,?,NULL)',rows)
        return db

    def test_test_first_then_val_and_safe_fill(self):
        db=self.build([('test_and_val',5,3,0),('existing_val',2,2,0),('other_train',100,1,1),
                       ('eligible_a',3,1,0),('eligible_b',4,1,0)])
        result=select_validation(db,5)
        routes=dict(db.execute('SELECT key,action FROM counts'))
        self.assertEqual(routes['test_and_val'],'excluded_test_overlap')
        self.assertEqual(routes['existing_val'],'val')
        self.assertEqual(routes['other_train'],'train')
        self.assertGreaterEqual(result['selected_validation_rows'],5)
        self.assertEqual(result['target_shortfall'],0)

    def test_quota_excess_and_shortfall_never_return_held_keys_or_remove_other_train(self):
        db=self.build([('existing_val_a',3,2,0),('existing_val_b',3,2,0),('other_train',100,1,1)])
        result=select_validation(db,2)
        self.assertEqual(result['selected_validation_rows'],3)
        self.assertEqual(Counter(x[0] for x in db.execute('SELECT action FROM counts')),
                         {'val':1,'excluded_val_quota':1,'train':1})
        db=self.build([('existing_val',1,2,0),('other_train',100,1,1)])
        result=select_validation(db,5)
        self.assertEqual(result['target_shortfall'],4)
        self.assertEqual(db.execute("SELECT action FROM counts WHERE key='other_train'").fetchone()[0],'train')


if __name__=='__main__':unittest.main()
