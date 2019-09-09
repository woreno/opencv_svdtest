

function axes_ = uiaxes(M, varargin)
        pos = [ 0 0 1 1];
        border = 0;
        if (numel(varargin)>0)
            if (strcmpi(varargin{1},'position'))
                pos = varargin{2};
                varargin{1} = [];
                varargin{2} = [];
                varargin = deleteEmptyCells(varargin);
            end
            if (strcmpi(varargin{1},'border'))
                border = varargin{2};
                varargin{1} = [];
                varargin{2} = [];
                varargin = deleteEmptyCells(varargin);
            end
        end
        
        dy = pos(4)/M(1);
        dx = pos(3)/M(2);
        for y=0: M(1)-1
            y0 = M(1)-y;
            for x=0: M(2)-1
                 axes_(y0,x+1) = axes('position',[pos(1)+x*dx+border, pos(2)+y*dy+border,  dx-border*2,  dy-border*2 ], varargin{:});
            end
        end

        set(axes_(:),'XTick',[], 'YTick',[],'YDir','reverse'); 

end