import gurobipy as grb
import time


def mip_callback(model, where):
    """
    :param model: model well constructed before optimize()
    :param where: just call in this way: model.optimize(mip_callback)
    :return: 
    """
    if where == grb.GRB.Callback.MIP:
        # obtain current objective value
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        # obtain current time
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)

        # obtain time record
        if 'last_time' not in mip_callback.__dict__:
            mip_callback.last_time = current_time
            mip_callback.last_obj = current_obj
            mip_callback.start_time = time.time()
            # objective value keeps unchanged for this mip_callback.time_limit: stop
            mip_callback.time_limit = 60

        # check whether objective value is unchanged
        # check every 10s
        if current_time - mip_callback.last_time > 5:
            # tolerance of difference between objective values during time_limit 
            if abs(current_obj - mip_callback.last_obj) < 1e-4:
                if time.time() - mip_callback.start_time > mip_callback.time_limit:
                    model.terminate() 
            else:
                mip_callback.last_time = current_time
                mip_callback.last_obj = current_obj
                mip_callback.start_time = time.time()  # objective value changes, reset time


# model = grb.Model()
# x = model.addVar(name="x")
# model.setObjective(x, grb.GRB.MAXIMIZE)
# model.addConstr(x <= 10)
# 
# model.setParam('TimeLimit', 3600)  
# 
# model.optimize(mip_callback)
