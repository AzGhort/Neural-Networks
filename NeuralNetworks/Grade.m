function answer = Grade(Scores, Boundaries)
% computes final grade of students based on their Scores and grades'
% Boundaries
if (nargin < 2)
    % default values for optional param
    Boundaries = [86,71,56];
end
answer = [];

for I=1:length(Scores)
    % find first boundary that is less than given score
    new = find(Boundaries<=Scores(I),1);
    if (isempty(new))
        % else the mark is 4
        new = 4;
    end
    answer = [answer;new];
end
end