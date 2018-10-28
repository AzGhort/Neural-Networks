function score = Score(StudentResults,MaxPoints,PointsWeight)
% computes final percentage score based on given StudentResults
if (nargin < 3)
    % default values for optional params
    MaxPoints = 75;
    PointsWeight = 35;
end
score = [];
% for all students
for i=1:size(StudentResults,2)
    % points from lessons
    s1 = sum(StudentResults{i}{1});
    if (s1 > MaxPoints)
        % maximal number of points from lessons
        s1 = MaxPoints;
    end
    % sum it all!
    sc = (s1/MaxPoints)*PointsWeight + sum(StudentResults{i}{2}) + StudentResults{i}{3};
    score = [score; sc];
end
end