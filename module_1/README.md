In this code, there are no tools yet.

The agent simply does the following:

Takes the user's task.
Sends it to the LLM.
Saves the LLM's response.
If the response contains "TASK_COMPLETE", it stops.
Otherwise, it sends "Continue" to the LLM.
Repeats the process.

This is the foundation of the agent loop.