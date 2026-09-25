*----------------------------------------------------------------------*
* COMPILE USING
*  f2py -m nmod -c thisfile.f	

        subroutine int_field_border(N_mp,xn,yn, bias_x,bias_y, dx,dy,
     +	efx, efy, Nxg, Nyg, Ex_n, Ey_n, inside_mat)
Cf2py intent(in)  N_mp   
Cf2py intent(in)  xn                                      
Cf2py intent(in)  yn   
Cf2py intent(in)  bias_x
Cf2py intent(in)  bias_y 
Cf2py intent(in)  dx
Cf2py intent(in)  dy
Cf2py intent(in)  efx
Cf2py intent(in)  efy
Cf2py intent(in)  inside_mat
Cf2py intent(in)  Nxg
Cf2py intent(in)  Nyg
Cf2py intent(out) Ex_n
Cf2py intent(out) Ey_n


        implicit none
        integer  N_mp
        real*8   xn(N_mp), yn(N_mp)
        real*8   bias_x, bias_y, dx, dy
        integer  Nxg,Nyg 
        real*8   efx(Nxg, Nyg), efy(Nxg, Nyg)
	    byte     inside_mat(Nxg, Nyg)
        integer  p
        real*8   fi, fj, hx, hy
        integer  i, j
        real*8   Ex_n(N_mp), Ey_n(N_mp)
        real*8   wei_ij,  wei_i1j, wei_ij1, wei_i1j1 
        real*8   fact_correct  
        logical anyexternal 

        do p=1,N_mp
        fi = 1+(xn(p)-bias_x)/dx;             !i index of particle's cell 
     	i  = int(fi);
     	hx = fi-dble(i);                      !fractional x position in cell
    

     	fj = 1+(yn(p)-bias_y)/dy;             !j index of particle' cell(C-like!!!!)
     	j = int(fj);
     	hy = fj-dble(j);                      !fractional y position in cell

     	anyexternal = .false.
        if (inside_mat(i, j)==0) then
            wei_ij =0.
            anyexternal = .true.
        else
            wei_ij = (1-hx)*(1-hy)
        end if
        
        if (inside_mat(i+1, j)==0) then
            wei_i1j =0.
            anyexternal = .true.
        else       
            wei_i1j = hx*(1-hy)
        end if        
        
        
        if (inside_mat(i, j+1)==0) then
            wei_ij1 =0.
            anyexternal = .true.
        else          
            wei_ij1 = (1-hx)*hy
        end if
            
        if (inside_mat(i+1, j+1)==0) then
            wei_i1j1 = 0.
            anyexternal = .true.
        else
            wei_i1j1  =  hx*hy  
        end if
        
     	!gather electric field
        if (i>0 .and. j>0 .and. i<Nxg .and. j<Nyg) then
            Ex_n(p) = efx((i),(j))*wei_ij;   
            Ex_n(p) = Ex_n(p) + efx((i+1),(j))*wei_i1j;
            Ex_n(p) = Ex_n(p) + efx((i),(j+1))*wei_ij1;
            Ex_n(p) = Ex_n(p) + efx((i+1),(j+1))*wei_i1j1;
        
            Ey_n(p) = efy((i),(j))*wei_ij;   
            Ey_n(p) = Ey_n(p) + efy((i+1),(j))*wei_i1j;
            Ey_n(p) = Ey_n(p) + efy((i),(j+1))*wei_ij1;
            Ey_n(p) = Ey_n(p) + efy((i+1),(j+1))*wei_i1j1;
        
            if (anyexternal .eqv. .true.) then
		if ((wei_ij+wei_i1j+wei_ij1+wei_i1j1)>0.) then
                fact_correct = 1./(wei_ij+wei_i1j+wei_ij1+wei_i1j1)
                Ex_n(p) = Ex_n(p)*fact_correct
                Ey_n(p) = Ey_n(p)*fact_correct
		else
		Ex_n(p) = 0.
                Ey_n(p) = 0.
		end if
            end if
        
        end if
        end do
        

     	
     	end subroutine

