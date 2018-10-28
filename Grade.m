function grade = Grade(Scores, Boundaries)
if (nargin < 2)
    Boundaries = [86,71,56];
end
answer = [];

for I=1:length(Scores)
    new = find(Boundaries<=Scores(I),1);
    if (isempty(new))
        new = length(Scores)+1;
    end
    answer = [answer new];
end

grade = answer';
end