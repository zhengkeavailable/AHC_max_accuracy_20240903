import gurobipy as grb
import time


def mip_callback(model, where):
    """
    :param model: model well constructed before optimize()
    :param where: just call in this way: model.optimize(mip_callback)
    :return: 
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)

        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            model.__dict__['start_time'] = time.time()
            model.__dict__['time_limit'] = 60

        # Check if objective value is unchanged
        if current_time - model.__dict__['last_time'] > 5:
            if abs(current_obj - model.__dict__['last_obj']) < 1e-4:
                if time.time() - model.__dict__['start_time'] > model.__dict__['time_limit']:
                    print("Terminating optimization due to stagnant objective value.")
                    model.terminate()
            else:
                # Reset the time record if objective value changes
                model.__dict__['last_time'] = current_time
                model.__dict__['last_obj'] = current_obj
                model.__dict__['start_time'] = time.time()


# model = grb.Model()
# x = model.addVar(name="x")
# model.setObjective(x, grb.GRB.MAXIMIZE)
# model.addConstr(x <= 10)
# 
# model.setParam('TimeLimit', 3600)  
# 
# model.optimize(mip_callback)
