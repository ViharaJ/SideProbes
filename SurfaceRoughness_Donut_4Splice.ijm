setBatchMode(true);
start = getTime();
sourcePath = "U:\\ROI\\Probes09-12 ROI";
subFolders_ofInterest = newArray("Probe09", "Probe10", "Probe11", "Probe12");

savePath = "C:\\Users\\v.jayaweera\\Pictures";
print(subFolders_ofInterest.length);

for(i = 0; i < subFolders_ofInterest.length; i++) //here i starts at 1 because you indicated that is the first folder of interest, but this could be any number
{
    openPath = sourcePath + "\\" + subFolders_ofInterest[i];
    
    files = getFileList(openPath);
    
    //Create save directoires
	mainSavePath = savePath + File.separator + subFolders_ofInterest[i];
	innerSurfaceSave = mainSavePath + File.separator + "Inner_Surface";
	
	outerS1 = mainSavePath + File.separator + "Outer_Surface_Upskin";
	outerS2 = mainSavePath + File.separator + "Outer_Surface_SideSkin_Right";
	outerS3 = mainSavePath + File.separator + "Outer_Surface_Downskin";
	outerS4 = mainSavePath + File.separator + "Outer_Surface_SideSkin_Left";
	
	File.makeDirectory(mainSavePath);
	File.makeDirectory(innerSurfaceSave);
	File.makeDirectory(outerS1);
	File.makeDirectory(outerS2);
	File.makeDirectory(outerS3);
	File.makeDirectory(outerS4);
		
    for(count = 0; count < files.length; count++) //here count should start at 0 in order to index through the folder properly (otherwise it won't start at the first file in the folder)
    {
    	open(openPath + File.separator + files[count]);
    	fullfileName = getTitle();
    	
    	fileName = fullfileName.substring(0, fullfileName.length-4);
		text = split(fileName, "-");
		
		probe = text[0];
		scale = text[3];
		slice = text[text.length-1];
		
		finalName =  probe + "-" + scale + "-" + slice + "-Outline";
    	
    	run("8-bit");
		x_max=getWidth();
		y_max=getHeight();
		
			
		run("Duplicate...", "duplicate range=4-4 use");
		print("Roughness analysis starts");
		
		setAutoThreshold("Default dark");
		//run("Threshold...");
		setAutoThreshold("Default dark");
		setOption("BlackBackground", false);
		run("Convert to Mask", "method=Default background=Dark");
		rename("Mask");
		run("Duplicate...", "duplicate");
		rename("OuterSurface");
		run("Fill Holes");
		run("Find Edges");
		run("Invert LUT");
		selectWindow("Mask");
		run("Find Edges");
		getTitle();
		print("returned title");
		imageCalculator("Subtract create", "Mask","OuterSurface");
		rename("InnerSurface");
		close("Mask");
		print("Seperated inner and outer surface");
		
		
		//Get all connected regions, then turn to gray scale (8-bit)
		run("Find Connected Regions", "allow_diagonal display_one_image regions_for_values_over=100 minimum_number_of_points=200 stop_after=-1");
		run("8-bit");
		run("Skeletonize (2D/3D)"); //Thin to one pixel
		save(innerSurfaceSave + "\\" +  finalName + ".tif");
		close("InnerSurface");
		
		selectWindow("OuterSurface");
		run("Find Connected Regions", "allow_diagonal display_one_image regions_for_values_over=100 minimum_number_of_points=200 stop_after=-1");
		run("8-bit");
		selectWindow("OuterSurface");
		run("Skeletonize (2D/3D)");
		
		
	
		//Get Top Quarter
		selectWindow("OuterSurface");
		run("Duplicate...", " ");
		makePolygon(0,0,x_max/2,y_max/2,x_max,0); // Polgon mit Drei Punkten: x1,y1,x2,y2,x3,y3
		run("Clear Outside");
		save(outerS1 + "\\" +  finalName + ".tif");
		
		
				
		//Get right quarter
		selectWindow("OuterSurface");
		run("Duplicate...", " ");
		makePolygon(x_max,0,x_max/2,y_max/2,x_max,y_max); // Polgon mit Drei Punkten: x1,y1,x2,y2,x3,y3
		run("Clear Outside");
		run("Rotate 90 Degrees Left");
		save(outerS2 + "\\" +  finalName + ".tif");
		
		
		//Get bottom quarter
		selectWindow("OuterSurface");
		run("Duplicate...", " ");
		makePolygon(x_max,y_max,x_max/2,y_max/2,0,y_max); // Polgon mit Drei Punkten: x1,y1,x2,y2,x3,y3
		run("Clear Outside");
		run("Rotate 90 Degrees Left");
		run("Rotate 90 Degrees Left");
		save(outerS3 + "\\" +  finalName + ".tif");
		
		
		//Get left quarter
		selectWindow("OuterSurface");
		run("Duplicate...", " ");
		makePolygon(0,y_max,x_max/2,y_max/2,0,0); // Polgon mit Drei Punkten: x1,y1,x2,y2,x3,y3
		run("Clear Outside");
		run("Rotate 90 Degrees Right");
		save(outerS4 + "\\" +  finalName + ".tif");
		close("*");
		
    }
}

setBatchMode(false);