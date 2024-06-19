Header header    # header information
int64 req        # request number
bool success     # True indicates LLM successfully responded
string info      # Human readable information
string response  # Response from the LLM
uint64 req_time  # Time prompt sent to LLM (nanosec)
uint64 res_time  # Time response from LLM recieved (nanosec)