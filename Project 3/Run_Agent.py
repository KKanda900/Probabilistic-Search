import sys, basic_agent_1, basic_agent_2, advanced_agent

'''
Agent Driver

Description: Runs the agents depending on command line arguments inputted.
'''

# initates the agent based on the requested command line argument
if __name__ == "__main__":
    # check if the user entered the right amount of arguments otherwise return failure
    if len(sys.argv) != 2:
        print("Usage Error:")
        print("Enter One of the following:")
        print("python Run_Agent.py ba1")
        print("python Run_Agent.py ba2")
        print("python Run_Agent.py adv")
        sys.exit(1)

    if sys.argv[1] == 'ba1':
        basic_agent_1.start_agent()
    elif sys.argv[1] == 'ba2':
        basic_agent_2.start_agent()
    else:
        advanced_agent.start_agent()
