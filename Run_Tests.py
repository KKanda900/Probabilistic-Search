import locationRelated2
import basic_agent1

if __name__ == "__main__":
    num_tests = 10
    f = open("Results.txt", "a")
    while num_tests != 0:
        results_ba2 = locationRelated2.run_agent_2()
        results_ba1 = basic_agent1.run_agent_1()
        f.write("Test {} Results: Basic Agent 1 Score - {} and Basic Agent 2 Score {}\n\n".format(num_tests, results_ba1, results_ba2)) 
        num_tests -= 1
    f.close()