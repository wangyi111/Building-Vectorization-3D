clear
clc

t = Tiff('sample_dsm_raw.tif','r');
H = read(t);

figure(1)
image(H)
hold on


S1 = shaperead('sample_polys_in.shp');

%figure(2)
for i=1:length(S1)
    r = S1(i).X;
    c = S1(i).Y;
    %x = c(1:(length(c)-2));
    %y = 1000-r(1:(length(r)-2));
    y = r(1:(length(r)-2));
    x = c(1:(length(c)-2));
    z = zeros(1,length(x));
    for j=1:length(r)-2
        left = max(1,r(j)-4);
        right = min(1000,r(j)+4);
        up = max(1,c(j)-4);
        down = min(1000,c(j)+4);
        window = H(left:right,up:down);
        z(j)=max(window(:));
    end
    patch(x,y,z,'red')
end


S2 = shaperead('sample_polys_out.shp');

%figure(2)
for i=1:length(S2)
    r = S2(i).X;
    c = S2(i).Y;
    %x = c(1:(length(c)-2));
    %y = 1000-r(1:(length(r)-2));
    y = r(1:(length(r)-2));
    x = c(1:(length(c)-2));
    z = zeros(1,length(x));
    for j=1:length(r)-2
        left = max(1,r(j)-4);
        right = min(1000,r(j)+4);
        up = max(1,c(j)-4);
        down = min(1000,c(j)+4);
        window = H(left:right,up:down);
        %z(j)=min(window(:));
        z(j)=50;
    end
    patch(x,y,z,'green')
end


for i=1:length(S2)
    
    r = S2(i).X;
    c = S2(i).Y;
    %x = c(1:(length(c)-2));
    %y = 1000-r(1:(length(r)-2));
    y = r(1:(length(r)-1));
    x = c(1:(length(c)-1));

    z1 = zeros(1,length(x));
    z2 = zeros(1,length(x));
    for j=1:length(z1)
        left = max(1,r(j)-4);
        right = min(1000,r(j)+4);
        up = max(1,c(j)-4);
        down = min(1000,c(j)+4);
        window = H(left:right,up:down);
        %z(j)=min(window(:));
        z1(j)=50;
        z2(j)=max(window(:));
    end    
    
    for j=1:length(x)-1
        x1 = x(j);
        y1 = y(j);
        x2 = x(j+1);
        y2 = y(j+1);
        zd1 = z1(j);
        zu1 = z2(j);
        zd2 = z1(j+1);
        zu2 = z2(j+1);
        
        lx = [x1,x2,x2,x1];
        ly = [y1,y2,y2,y1];
        lz = [zu1,zu2,zd2,zd1];
        
        patch(lx,ly,lz,'blue')
    end
    
    

end



%axis([1,1000,1,1000,0,70])
view(3)

