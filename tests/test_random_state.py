import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import unittest
from arfpy import arf

# reproducibility through the random_state argument of arf() and forge()
# iris with a categorical target covers both the continuous and the categorical sampling paths

class TestRandomState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        iris = load_iris()
        cls.df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
        cls.df['target'] = pd.Series(iris['target']).astype('category')

    def fit(self, seed):
        model = arf.arf(x = self.df, random_state = seed, verbose = False)
        model.forde()
        return model

    # the same seed gives the same forest, the same density and the same synthetic data
    def test_same_seed_same_output(self):
        a, b = self.fit(509), self.fit(509)
        self.assertEqual(a.acc, b.acc, 'discriminator accuracies differ for the same seed')
        pd.testing.assert_frame_equal(a.params.reset_index(drop = True), b.params.reset_index(drop = True))
        pd.testing.assert_frame_equal(a.forge(n = 50), b.forge(n = 50))

    # different seeds give different synthetic data
    def test_different_seed_different_output(self):
        a, b = self.fit(509), self.fit(510)
        self.assertFalse(a.forge(n = 50).equals(b.forge(n = 50)), 'different seeds produced identical data')

    # a seed passed to forge() overrides the model's random state for that call only
    def test_forge_seed(self):
        model = self.fit(509)
        pd.testing.assert_frame_equal(model.forge(n = 50, random_state = 1), model.forge(n = 50, random_state = 1))
        self.assertFalse(model.forge(n = 50).equals(model.forge(n = 50)), 'successive forge() calls repeated themselves')

    # without a seed, numpy's global random state still controls everything, as before
    def test_global_seed_still_works(self):
        np.random.seed(509)
        a = arf.arf(x = self.df, verbose = False); a.forde(); out_a = a.forge(n = 50)
        np.random.seed(509)
        b = arf.arf(x = self.df, verbose = False); b.forde(); out_b = b.forge(n = 50)
        pd.testing.assert_frame_equal(out_a, out_b)

if __name__ == '__main__':
    unittest.main()
