function retval = randv2n(n1,S1,R1,n2,S2,R2)
   first = generateRandomData(1,n1,S1,R1);
   second = generateRandomData(1,n2,S2,R2);
   
   val=[first,second];
   retval=val(randperm(n1+n2));
end
