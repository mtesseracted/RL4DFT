"""The feature-4 dashboard: observer panels plus the human endpoint.

Imports no physics module. It renders what the record store contains and
nothing else -- if a number is on the screen, it is in a record or reachable
from one (D13). That rule is what keeps the dashboard honest: a panel cannot
quietly compute a quantity the search never measured and present it beside
ones it did.
"""
