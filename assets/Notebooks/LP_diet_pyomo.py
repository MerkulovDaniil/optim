# Creating pyomo model
from pyomo.environ import *
infinity = float('inf')

model = AbstractModel()

# Foods (products)
model.p = Set()
# Nutrients
model.n = Set()

# Cost of each food
model.c    = Param(model.p, within=PositiveReals)

# Amount of nutrient in each food
model.W    = Param(model.p, model.n, within=NonNegativeReals)

# Lower and upper bound on each nutrient
model.Nmin = Param(model.n, within=NonNegativeReals, default=0.0)
model.Nmax = Param(model.n, within=NonNegativeReals, default=infinity)

# Defining variables
# Number of servings consumed of each food
model.x = Var(model.p, within=NonNegativeReals) 

# Defining cost function
# Minimize the cost of food that is consumed
def cost_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.p)
model.cost = Objective(rule=cost_rule)

# Defining the constraints
# Limit nutrient consumption for each nutrient
def nutrient_rule(model, j):
    value = sum(model.W[i,j]*model.x[i] for i in model.p)
    return model.Nmin[j] <= value <= model.Nmax[j]
model.nutrient_limit = Constraint(model.n, rule=nutrient_rule)
