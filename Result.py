import advanced_agent
import random

if __name__ == "__main__":

    f = open("Results1.txt", "a")

    trials = 10
    while trials != 0:

        ba1_map = test2.AgentClass(50)

        num = 1
        ba1_results = []
        target_terrain = ba1_map.target_info.terrain_location
        x = 0
        y = 0
        while num != 11:
            score1 = ba1_map.start_agent(50)
            ba1_results.append(score1)
            num += 1
        f.write("Target Terrain {} Basic Agent 1 Score {}\n\n".format(target_terrain, (sum(ba1_results)/len(ba1_results))))
        trials -= 1