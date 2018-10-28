function M = randn2d(p,show)
  if nargin < 2
    show = false
  end
  % Init
  totalN = sum(p(1,:));
  M = zeros(2, totalN);
  index = 1;
  % Generate clusters
  for cluster = 1:size(p,2)
    count = p(1,cluster);
    x = sqrt(p(4, cluster))*randn(1,count)+p(2, cluster);
    y = sqrt(p(5, cluster))*randn(1,count)+p(3, cluster);
    M(1:2,index:index+count-1)=[x;y];
    index = index+count;
    if show
      plot(x,y,'x');
      hold on;
    end
  end
  % Shuffle
  tmp = randperm(size(M,2));
  M = M(:,tmp);
  % Plot graph
  if show
    hold off;
  end
end

        