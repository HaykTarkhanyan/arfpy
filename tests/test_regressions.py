import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import unittest
from arfpy import arf

# regression tests for leaf coverage in forde() and for forde(oob = True)

class TestRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(509)
        cls.df = pd.DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)

    def leaf_coverage(self, model):
        return model.bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()

    # min_node_size = 1 leaves single-observation leaves, whose coverage forde() sets to zero; forge() used to
    # draw leaves with probabilities that no longer summed to one (issue #4)
    def test_forge_with_single_observation_leaves(self):
        model = arf.arf(x = self.df, min_node_size = 1, verbose = False)
        model.forde()
        sums = self.leaf_coverage(model).groupby('tree')['cvg'].sum()
        self.assertTrue(np.allclose(sums, 1), 'leaf coverage does not sum to one within each tree')
        gen = model.forge(n = 100)
        self.assertEqual(gen.shape, (100, self.df.shape[1]))
        self.assertFalse(gen.isna().any().any(), 'generated data contains missing values')

    # forde(oob = True) referenced an attribute that does not exist
    def test_forde_oob(self):
        model = arf.arf(x = self.df, verbose = False)
        model.forde(oob = True)
        sums = self.leaf_coverage(model).groupby('tree')['cvg'].sum()
        self.assertTrue(np.allclose(sums, 1), 'out-of-bag leaf coverage does not sum to one within each tree')
        gen = model.forge(n = 100)
        self.assertEqual(gen.shape, (100, self.df.shape[1]))

    # with oob = True, each tree's parameters may only use observations that tree did not draw
    def test_forde_oob_uses_only_out_of_bag_rows(self):
        model = arf.arf(x = self.df, num_trees = 5, verbose = False)
        model.forde(oob = True)
        n_real = self.df.shape[0]
        leaves = model.clf.apply(model.x_real)
        for tree in range(model.num_trees):
            oob = np.setdiff1d(np.arange(n_real), model.clf.estimators_samples_[tree])
            used = model.bnds.loc[model.bnds['tree'] == tree, 'nodeid'].unique()
            self.assertTrue(set(used) <= set(leaves[oob, tree]), 'a leaf without out-of-bag observations was kept')

if __name__ == '__main__':
    unittest.main()
