function ret=HowManyPercent(TargetGrade,StudentResults,Boundaries,MaxPoints,PointsWeight,MaxExamScore)
if (nargin < 6)
    % default values for optional params
    Boundaries = [86,71,56];
    MaxPoints = 75;
    PointsWeight = 35;
    MaxExamScore = 45;
end
ret=[];
for i=1:size(TargetGrade,1)
    if (TargetGrade(i) < 4)
        % points we need
        targetPoints = Boundaries(TargetGrade(i));
    else
        % do not need any points not to pass
        targetPoints = 0;
    end
    % points from lessons
    s1 = sum(StudentResults{i}{1});
    if (s1 > MaxPoints)
        % maximal number of points from lessons
        s1 = MaxPoints;
    end
    % points before the exam
    actualPoints = (s1/MaxPoints)*PointsWeight + sum(StudentResults{i}{2});
    pointsNeeded = targetPoints-actualPoints;
    if (pointsNeeded > MaxExamScore)
        % cant get the target grade :(
        pointsNeeded = NaN;
    elseif (pointsNeeded < 0)
        % already got the target grade :)
        pointsNeeded = 0;
    end
    ret=[ret;pointsNeeded];     
end
end