#!/usr/bin/env expect
set images [glob -directory "predicted_images" -- "*.jpg"]
foreach image $images {
  puts "predicted_images/$image"
}
spawn ./darknet pibbles cfg/yolov3.cfg yolov3.weights

expect "Enter Image Path:"
send "data/dog.jpg\r"
expect "Person"
