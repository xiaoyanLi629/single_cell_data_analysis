function [coordinates] = sphere_creation(mean, std)
    
    dim = length(mean);
    rng(0,'twister');
    rvals = 2*rand(dim,1)-1;
    elevation = asin(rvals);

    azimuth = 2*pi*rand(dim,1);

    radii = ones(dim, 1);
    radii = radii*100+10;
    [x,y,z] = sph2cart(azimuth,elevation,radii);
    s = randi([100, 300], dim, 1);
    c = rand(dim, 3);
    coordinates = [x, y, z];
    figure
    scatter3(x, y, z, s, c, '.')
    line(x, y, z);
    axis equal
    
end

% A = rand(100, 2);
% A(:, 1) = 1*cos(A(:, 1)*pi/2).*cos(A(:, 2)*pi/2);
% A(:, 2) = 1*cos(A(:, 1)*pi/2).*sin(A(:, 2)*pi/2);
% A(:, 3) = 1*sin(A(:, 1));
% scatter3(A(:, 1), A(:, 2), A(:, 3), '.');

% theta=linspace(0,2*pi,100);
% phi=linspace(0,pi,80);
% [theta,phi]=meshgrid(theta,phi);
% rho=1;
% x=rho*sin(phi).*cos(theta);
% y=rho*sin(phi).*sin(theta);
% z=rho*cos(phi);
% figure
% surf(x,y,z)
% axis equal