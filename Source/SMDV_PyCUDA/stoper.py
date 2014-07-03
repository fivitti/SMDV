#-*- coding:utf-8 -*-

from time import time

class Timer:
    def __init__(self):
        self._elapsed  = 0.0
        self._starttime = time()
        self._started = False

    def start(self):
        self._starttime = time()
        self._started = True

    def stop(self):
        self._elapsed += (time() - self._starttime)
        self._started = False

    def reset(self):
        self._elapsed = 0.0

    def get_elapsed(self):
        if self._started:
            return (self._elapsed + time() - self._starttime)
        else:
            return self._elapsed

    def get_result(self):
      # zwraca tuple: (hours, minutes, seconds, microseconds)
      time = 0
      if self._started:
        time = (self._elapsed + time() - self._starttime)
      else:
        time = self._elapsed

      seconds = time
      minutes = seconds // 60
      seconds = round((seconds % 60),3)
      hours = minutes // 60
      minutes = minutes % 60
      mseconds = round((time * 1000), 4)
      return (hours, minutes, seconds, mseconds)



if __name__ == '__main__':
  _timer = Timer()
  _timer.start()

  ###............ tutaj kod do wykonania

  _timer.stop()

  print _timer.get_elapsed()
  print _timer.get_result()

  ## really format round - bug?
  print str(_timer.get_result()[3]) + " ms"
  _timer.reset()