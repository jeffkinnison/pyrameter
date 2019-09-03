"""Structured hyperparameter optimization with flexible methods and storage."""

from .optimizer import FMin
from .domains import *


def uniform(loc, scale):
    return ContinuousDomain('uniform', loc=loc, scale=scale)


def normal(mu, sigma):
    return ContinuousDomain('normal', loc=mu, scale=sigma)
