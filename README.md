# Task List

You may do these tasks in any order, but take note that they are listed in the order your team has prioritized completing them.

Reminder that you are NOT expected to complete all tasks. You are expected to write clean, readable code. Remember to add comments explaining what you were working on if you run out of time in the middle of a task.


## Task 1

We have decided to move forward with an interest-based approach to matching `Volunteers` with `Tasks`. 

a. Implement the method with the following prototype in assignment_server.py:

**get_interested_volunteers(self, task)**

This method should return a list of all `Volunteer` objects which have the given `Task` in their `interested_tasks` lists. If no volunteers list the task, it should return an empty list.

b. Additionally, we want to be able to compare tasks based on an objective measure of desirability across volunteers. We have decided to use a simple calculation for this purpose: a task's desirability score for a given volunteer is calculated by the function `1 / (x + 1)`, where x is the index of the task in the volunteer's `interested_tasks` list.

Implement the method with the following prototype in volunteer.py:

**get_task_desirability_score(self, task)**

This method should return the given `Task`'s desirability score for the current `Volunteer`, by simply implementing the function described above. If the `Task` does not appear in the `Volunteer`'s interested_tasks list, the method should return `0`.


## Task 2

A `Task`'s overall desirability is the sum of its desirability scores for all `Volunteers` who have included the `Task` in their interested_tasks lists. Implement the method with the following prototype in assignment_server.py:

**get_tasks_by_desirability(self)**

This method should return a list of `Tasks` sorted from greatest to least overall desirability, and with people-facing `Tasks` listed before the other `Tasks`.

We recently conducted internal research that shows that volunteers are more satisfied with their work and more likely to volunteer again when they are given a people-facing task. Moving people-facing tasks directly to the front of the desirability list ensures that we distribute those tasks effectively.


## Task 3

Some volunteers have expressed dissatisfaction with the tasks they've been assigned. We want to track and increase volunteer satisfaction, so we have established a Volunteer Satisfaction Score. You can call `get_volunteer_satisfaction_score()` from util.py to determine the score for current assignments.

In assignment_server.py, implement the method with the following prototype:

**assign_tasks_improved(self)**

In this function, suggest and implement changes so that our assignment server yields higher volunteer satisfaction than it does with the `assign_tasks(self)` function as currently implemented. Note that we are not looking for a perfect solution; our clients would prefer one or two completed minor improvements to the current algorithm over an "optimal" solution that is buggy or incomplete.

You are welcome to add any helper methods, implement any algorithm and/or use any underlying data structures you like, but you are encouraged to make sure your decisions are well documented and your code is appropriately decomposed.

# experimental_attention_free_diff
