#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import datetime

def publish_timestamp():
    rospy.init_node('timestamp_publisher', anonymous=True)
    publisher = rospy.Publisher('agent_environment', String, queue_size=10)
    rate = rospy.Rate(5)  # Frequency of 1 Hz
    
    while not rospy.is_shutdown():
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rospy.loginfo(f"Publishing current timestamp: {current_time}")
        publisher.publish(current_time)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_timestamp()
    except rospy.ROSInterruptException:
        pass
