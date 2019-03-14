import pulp

model = pulp.LpProblem("Profit maximising problem", pulp.LpMaximize)

A = pulp.LpVariable('A', lowBound=0, cat='Integer')
B = pulp.LpVariable('B', lowBound=0, cat='Integer')


model += 30000 * A + 45000 * B, "Profit"

model += 3 * A + 4 * B <= 30
model += 5 * A + 6 * B <= 60
model += 1.5 * A + 3 * B <= 21

model.solve()
pulp.LpStatus[model.status]

print("Production of Car A = {}".format(A.varValue))
print("Production of Car B = {}".format(B.varValue))

print(pulp.value(model.objective))