import basic_agent_1
import basic_agent_2
import random

if __name__ == "__main__":
    """ f = open("Results2.txt", "a")
    
    ba2_map = basic_agent_2.Basic_Agent_2(50)

    num = 1
    ba2_results = []
    target_terrain = ba2_map.target_info.terrain_location
    while num != 11:
        x = random.randint(0, 49)
        y = random.randint(0, 49)
        score = ba2_map.start_agent(x, y)
        ba2_results.append(score)
        num+=1

    f.write("Basic Agent 2 Score for Target in {}: {}\n\n".format(target_terrain, (sum(ba2_results)/len(ba2_results))))

    f.close() """

    f = open("Results1.txt", "a")

    ba2_map = basic_agent_2.Basic_Agent_2(50)

    num = 1
    ba2_results = []
    target_terrain = ba2_map.target_info.terrain_location
    while num != 11:
        x = random.randint(0, 49)
        y = random.randint(0, 49)
        score = ba2_map.start_agent(x, y)
        ba2_results.append(score)
        num+=1

    f.write("Basic Agent 2 Score for Target in {}: {}\n\n".format(
        target_terrain, (sum(ba2_results)/len(ba2_results))))
