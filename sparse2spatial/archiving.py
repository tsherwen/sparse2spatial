"""

Make high-res and re-gridded data files for archiving

"""


# ---------------------------------------------------------------------------
# ------------------------- Regridding of output -------------------------
# ---------------------------------------------------------------------------
def mk_NetCDF_files_for_data_archiving(target='iodide'):
    """
    Make data into NetCDF files for archiving at a data centre
    """
    # Main output excludes Skaggerak data
    rm_Skagerrak_data=True
    # Manually set version
    version = 'v0.0.1'
    # Master output file to regrid.
    if rm_Skagerrak_data:
        ext_str = '_No_Skagerrak'
    else:
        ext_str = ''
    file2regrid = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'.format(target, ext_str)
    folder = get_file_locations('data_root')
    dsA = xr.open_dataset(folder + file2regrid)
    # Make sure there are not spaces in variable names
    dsA = add_attrs2target_ds( dsA, add_varname_attrs=False, add_global_attrs=False,
                               update_varnames_to_remove_spaces=True   )
    # remove existing parameters if they are there
    try:
        del dsA['Chance2014_STTxx2_I']
    except KeyError:
        pass
    try:
        del dsA['MacDonald2014_iodide']
    except KeyError:
        pass
    # ( now: nOutliers + nSkagerak )
    RFR_dict = build_or_get_current_models(rm_Skagerrak_data=rm_Skagerrak_data)
#    RFR_dict = build_or_get_current_models( rm_Skagerrak_data=False )
    topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)
    # hardcore v0.0.0 / v0.0.1 models (as saved as Python 2 pickles. )
#    topmodels = ['RFR(TEMP+DEPTH+SAL+Phos)', 'RFR(TEMP+DEPTH+NO3+SWrad)', 'RFR(TEMP+DEPTH+SAL+NO3)', 'RFR(TEMP+DEPTH+SAL+SWrad)', 'RFR(TEMP+DEPTH+SAL+ChlrA)', 'RFR(TEMP+DEPTH+SAL)', 'RFR(TEMP+SWrad+NO3+MLD+SAL)', 'RFR(TEMP+NO3+MLD+SAL)', 'RFR(TEMP+DEPTH+NO3)', 'RFR(TEMP+DEPTH+ChlrA)']
    # - Now regrid the output
    regrid_output_to_common_res_as_NetCDFs( dsA=dsA, topmodels=topmodels )

    # - Now save the high res main files
    # LWI's '_FillValue' key is already set, so remove this.
    try:
        del dsA['LWI'].attrs['_FillValue']
    except:
        pass
    # what are the variables that should be in all files?
    standard_vars = ['Ensemble_Monthly_mean', 'Ensemble_Monthly_std', 'LWI']
    # Also save the main file with just the main prediction
    name2save = 'predicted_{}_0.125x0.125_Ns_Just_Ensemble'.format(target)
    dsA[standard_vars].to_netcdf(name2save+'.nc')

    # Add with ensemble members
    name2save = 'predicted_{}_0.125x0.125_Ns_All_Ensemble_members'.format(target)
    dsA[standard_vars+topmodels].to_netcdf(name2save+'.nc')

    # - Also save out the file that is inc. excluded data.
    name2save = 'predicted_{}_0.125x0.125_All_Ensemble_members'.format(target)
    #
    ext_str = ''
    file2regrid = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'.format(target, ext_str)
    folder = get_file_locations('data_root')
    dsA2 = xr.open_dataset( folder+file2regrid )
    # Make sure there are not spaces in variable names
    dsA2 = add_attrs2target_ds( dsA2, add_varname_attrs=False,
                                add_global_attrs=False,
                                update_varnames_to_remove_spaces=True   )
    # Add LWI to array
    try:
        dsA2['LWI']
    except KeyError:
        dsA2 = add_LWI2array(dsA2, res='0.125x0.125',
                             inc_booleans_and_area=False)
    # LWI's '_FillValue' key is already set, so remove this.
    try:
        del dsA2['LWI'].attrs['_FillValue']
    except:
        pass
    # Save NetCDF
    dsA2[standard_vars+topmodels].to_netcdf(name2save+'.nc')




def regrid_output_to_common_res_as_NetCDFs(topmodels=None, target='iodide',
                                           rm_Skagerrak_data=False, dsA=None,
                                           just_1x1_grids=False, debug=False):
    """ Regrid output various common model resolutsion """
    import xesmf as xe
    # Get file and location to regrid
    if rm_Skagerrak_data:
        ext_str = '_No_Skagerrak'
    else:
        ext_str = ''
    if isinstance(dsA, type(None)):
        file2regrid = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'.format(target, ext_str)
        folder = get_file_locations('data_root')
        dsA = xr.open_dataset(folder + file2regrid)
    # Add LWI to array
    try:
        dsA['LWI']
    except KeyError:
        dsA = add_LWI2array(dsA, res='0.125x0.125',
                            inc_booleans_and_area=False)
    # Which grids should be regridded too?
    grids = reses2regrid2(just_1x1_grids=just_1x1_grids)
    vars2regrid = list(dsA.data_vars)
    # Remove any models?
    if not isinstance(topmodels, type(None)):
        # remove the RFRs that are not in the topmodels list
        vars2pop = []
        for var2use in vars2regrid:
            if ('RFR' in var2use):
                if (var2use not in topmodels):
                    vars2pop += [ vars2regrid.index(var2use) ]
#                    vars2regrid.pop(vars2regrid.index(var2use))
                    if debug:
                        print('Deleting var:',  var2use, vars2regrid.index(var2use) )
        # Now remove by pop
        [vars2regrid.pop(i) for i in sorted(vars2pop)[::-1] ]

    # --- Regrid output
    for grid in grids.keys():
        # Create a dataset to re-grid into
        ds_out = xr.Dataset({
            #			'time': ( ['time'], dsA['time'] ),
            'lat': (['lat'], grids[grid]['lat']),
            'lon': (['lon'], grids[grid]['lon']),
        })
        # Create a regidder (to be reused )
        regridder = xe.Regridder(dsA, ds_out, 'bilinear', reuse_weights=True)
#		regridder  # print basic regridder information.
        # loop and regrid variables
        ds_l = []
        for var2use in vars2regrid:
            # create a dataset to re-grid into
            ds_out = xr.Dataset({
                #				'time': ( ['time'], dsA['time'] ),
                'lat': (['lat'], grids[grid]['lat']),
                'lon': (['lon'], grids[grid]['lon']),
            })
            # get a DataArray
            dr = dsA[var2use]
            # build regridder
            dr_out = regridder(dr)
            # Important note: Extra dimensions must be on the left, i.e. (time, lev, lat, lon) is correct but (lat, lon, time, lev) would not work. Most data sets should have (lat, lon) on the right (being the fastest changing dimension in the memory). If not, use DataArray.transpose or numpy.transpose to preprocess the data.
            # exactly the same as input
            xr.testing.assert_identical(dr_out['time'], dsA['time'])
            # save variable
            ds_l += [dr_out]
        # Combine variables
#        ds = xr.concat( ds_l )
        ds = xr.Dataset()
        for n, var2use in enumerate(vars2regrid):
            ds[var2use] = ds_l[n]
            # Add atributes
            Vars2NotRename = 'LWI', 'LonghurstProvince'
            if var2use not in Vars2NotRename:
                ds = add_attrs2target_ds(ds, add_global_attrs=False,
                                         varname=var2use)
            else:
                # Update attributes too
                attrs = ds['LWI'].attrs.copy()
                attrs['long_name'] = 'Land/Water/Ice index'
                attrs['Detail'] = 'A Land-Water-Ice mask. It is 1 over continental areas, 0 over open ocean, and 2 over seaice covered ocean.'
                ds['LWI'].attrs = attrs
        # clean up
        regridder.clean_weight_file()
        # Make sure the file has appropriate attributes
        ds = add_attrs2target_ds(ds, add_varname_attrs=False)
        # Time values
        ds = update_time_in_NetCDF2save(ds)
        # Save the file
        filename = 'Oi_prj_output_{}_field_{}'.format(target, grid)
        filename = AC.rm_spaces_and_chars_from_str(filename)
        ds.to_netcdf(filename+'.nc')


def reses2regrid2(just_1x1_grids=False):
    """ Func. to store resolutions and lat and lons to use """
    # --- LAT, LON dictionary
    use_v_0_0_0_grids = False
    use_v_0_0_1_grids = True
    if use_v_0_0_0_grids:
        d = {
        # BASE
        #  GEOSChem in GEOS5 \citep{Hu2017_ACPD}
        # 0.25 GEOS5
        '0.25x0.25_deg_centre_GENERIC': {
            'lon': np.arange(-180.125, 180.125, 0.25),
            'lat': np.arange(-90.125, 90.125, 0.25)
        },
        # Do a 0.5 degree grid for the LongHurst Provinces
        '0.5x0.5_deg_centre_GENERIC': {
            'lon': np.arange(-180.5, 180.5, 0.5),
            'lat': np.arange(-90.5, 90.5, 0.5)
        },
        # Do a 1x1 degree grid
        '1x1_deg_centre_GENERIC': {
            'lon': np.arange(-180.5, 180.5, 1),
            'lat': np.arange(-90.5, 90.5, 1)
        },
        # Do a 1x1 degree grid (centered on 0.5)
        '1x1_deg_0.5_centre_GENERIC': {
            'lon': np.arange(-180, 181, 1),
            'lat': np.arange(-90, 90, 1),
        },
        # GISS ModelE (Miller et al., 2014)
        '2x2.5_deg_centre_GISS': {
            'lon': np.arange(-178.75, 182.75, 2.5),
            'lat': np.arange(-90, 90, 2)
        },
        # ACCMIP (Lamarque et al., 2013)
        '2x2_deg_centre_ACCMIP': {
            'lon': np.arange(-180, 180, 5),
            'lat': np.arange(-90, 90, 4)
        },
        # GEOSChem (Bey et al., 2001) - 4◦ x5◦
        '2x2.5_deg_centre_GEOSChem': {
            'lon': np.arange(-181.25, 181.25, 2.5),
            'lat': np.arange(-91, 91, 2)
        },
        # UKCA (O’Connor et al., 2014)
        '2x3.75_deg_centre_UKCA': {
            'lon': np.arange(-180, 180, 5),
            'lat': np.arange(-90, 90, 4)
        },
        # GEOSChem (Bey et al., 2001) - 4◦ x5◦
        '4x5_deg_centre_GEOSChem': {
            'lon': np.arange(-182.5, 182, 5),
            'lat': np.arange(-92, 90, 4)
        },
        }
    elif (use_v_0_0_1_grids):
        d = {
        # BASE
        #  GEOSChem in GEOS5 \citep{Hu2017_ACPD}
        # 0.25 GEOS5
        '0.25x0.25_deg_centre_GENERIC': {
            'lon': np.array([-180+(i*0.25) for i in np.arange((360./0.25)-1)]),
            'lat': np.array([-90+(i*0.25) for i in np.arange((180./0.25)+1)])
        },
        # Do a 0.5 degree grid for the LongHurst Provinces
        '0.5x0.5_deg_centre_GENERIC': {
            'lon': np.array([-180+(i*0.5) for i in np.arange((360./0.5)-1)]),
            'lat': np.array([-90+(i*0.5) for i in np.arange((180./0.5)+1)])
        },
        # Do a 1x1 degree grid
        '1x1_deg_centre_GENERIC': {
            'lon': np.array([
       -179.5, -178.5, -177.5, -176.5, -175.5, -174.5, -173.5, -172.5,
       -171.5, -170.5, -169.5, -168.5, -167.5, -166.5, -165.5, -164.5,
       -163.5, -162.5, -161.5, -160.5, -159.5, -158.5, -157.5, -156.5,
       -155.5, -154.5, -153.5, -152.5, -151.5, -150.5, -149.5, -148.5,
       -147.5, -146.5, -145.5, -144.5, -143.5, -142.5, -141.5, -140.5,
       -139.5, -138.5, -137.5, -136.5, -135.5, -134.5, -133.5, -132.5,
       -131.5, -130.5, -129.5, -128.5, -127.5, -126.5, -125.5, -124.5,
       -123.5, -122.5, -121.5, -120.5, -119.5, -118.5, -117.5, -116.5,
       -115.5, -114.5, -113.5, -112.5, -111.5, -110.5, -109.5, -108.5,
       -107.5, -106.5, -105.5, -104.5, -103.5, -102.5, -101.5, -100.5,
        -99.5,  -98.5,  -97.5,  -96.5,  -95.5,  -94.5,  -93.5,  -92.5,
        -91.5,  -90.5,  -89.5,  -88.5,  -87.5,  -86.5,  -85.5,  -84.5,
        -83.5,  -82.5,  -81.5,  -80.5,  -79.5,  -78.5,  -77.5,  -76.5,
        -75.5,  -74.5,  -73.5,  -72.5,  -71.5,  -70.5,  -69.5,  -68.5,
        -67.5,  -66.5,  -65.5,  -64.5,  -63.5,  -62.5,  -61.5,  -60.5,
        -59.5,  -58.5,  -57.5,  -56.5,  -55.5,  -54.5,  -53.5,  -52.5,
        -51.5,  -50.5,  -49.5,  -48.5,  -47.5,  -46.5,  -45.5,  -44.5,
        -43.5,  -42.5,  -41.5,  -40.5,  -39.5,  -38.5,  -37.5,  -36.5,
        -35.5,  -34.5,  -33.5,  -32.5,  -31.5,  -30.5,  -29.5,  -28.5,
        -27.5,  -26.5,  -25.5,  -24.5,  -23.5,  -22.5,  -21.5,  -20.5,
        -19.5,  -18.5,  -17.5,  -16.5,  -15.5,  -14.5,  -13.5,  -12.5,
        -11.5,  -10.5,   -9.5,   -8.5,   -7.5,   -6.5,   -5.5,   -4.5,
         -3.5,   -2.5,   -1.5,   -0.5,    0.5,    1.5,    2.5,    3.5,
          4.5,    5.5,    6.5,    7.5,    8.5,    9.5,   10.5,   11.5,
         12.5,   13.5,   14.5,   15.5,   16.5,   17.5,   18.5,   19.5,
         20.5,   21.5,   22.5,   23.5,   24.5,   25.5,   26.5,   27.5,
         28.5,   29.5,   30.5,   31.5,   32.5,   33.5,   34.5,   35.5,
         36.5,   37.5,   38.5,   39.5,   40.5,   41.5,   42.5,   43.5,
         44.5,   45.5,   46.5,   47.5,   48.5,   49.5,   50.5,   51.5,
         52.5,   53.5,   54.5,   55.5,   56.5,   57.5,   58.5,   59.5,
         60.5,   61.5,   62.5,   63.5,   64.5,   65.5,   66.5,   67.5,
         68.5,   69.5,   70.5,   71.5,   72.5,   73.5,   74.5,   75.5,
         76.5,   77.5,   78.5,   79.5,   80.5,   81.5,   82.5,   83.5,
         84.5,   85.5,   86.5,   87.5,   88.5,   89.5,   90.5,   91.5,
         92.5,   93.5,   94.5,   95.5,   96.5,   97.5,   98.5,   99.5,
        100.5,  101.5,  102.5,  103.5,  104.5,  105.5,  106.5,  107.5,
        108.5,  109.5,  110.5,  111.5,  112.5,  113.5,  114.5,  115.5,
        116.5,  117.5,  118.5,  119.5,  120.5,  121.5,  122.5,  123.5,
        124.5,  125.5,  126.5,  127.5,  128.5,  129.5,  130.5,  131.5,
        132.5,  133.5,  134.5,  135.5,  136.5,  137.5,  138.5,  139.5,
        140.5,  141.5,  142.5,  143.5,  144.5,  145.5,  146.5,  147.5,
        148.5,  149.5,  150.5,  151.5,  152.5,  153.5,  154.5,  155.5,
        156.5,  157.5,  158.5,  159.5,  160.5,  161.5,  162.5,  163.5,
        164.5,  165.5,  166.5,  167.5,  168.5,  169.5,  170.5,  171.5,
        172.5,  173.5,  174.5,  175.5,  176.5,  177.5,  178.5,  179.5]),
            'lat': np.array([
       -89.5, -88.5, -87.5, -86.5, -85.5, -84.5, -83.5, -82.5, -81.5,
       -80.5, -79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5,
       -71.5, -70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5,
       -62.5, -61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5,
       -53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5,
       -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5,
       -35.5, -34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5,
       -26.5, -25.5, -24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5,
       -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5,  -9.5,
        -8.5,  -7.5,  -6.5,  -5.5,  -4.5,  -3.5,  -2.5,  -1.5,  -0.5,
         0.5,   1.5,   2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,
         9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,
        18.5,  19.5,  20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,
        27.5,  28.5,  29.5,  30.5,  31.5,  32.5,  33.5,  34.5,  35.5,
        36.5,  37.5,  38.5,  39.5,  40.5,  41.5,  42.5,  43.5,  44.5,
        45.5,  46.5,  47.5,  48.5,  49.5,  50.5,  51.5,  52.5,  53.5,
        54.5,  55.5,  56.5,  57.5,  58.5,  59.5,  60.5,  61.5,  62.5,
        63.5,  64.5,  65.5,  66.5,  67.5,  68.5,  69.5,  70.5,  71.5,
        72.5,  73.5,  74.5,  75.5,  76.5,  77.5,  78.5,  79.5,  80.5,
        81.5,  82.5,  83.5,  84.5,  85.5,  86.5,  87.5,  88.5,  89.5]),
        },
        # Do a 1x1 degree grid (centered on 0.5)
        '1x1_deg_0.5_centre_GENERIC': {
            'lon': np.array([
       -180, -179, -178, -177, -176, -175, -174, -173, -172, -171, -170,
       -169, -168, -167, -166, -165, -164, -163, -162, -161, -160, -159,
       -158, -157, -156, -155, -154, -153, -152, -151, -150, -149, -148,
       -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137,
       -136, -135, -134, -133, -132, -131, -130, -129, -128, -127, -126,
       -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
       -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104,
       -103, -102, -101, -100,  -99,  -98,  -97,  -96,  -95,  -94,  -93,
        -92,  -91,  -90,  -89,  -88,  -87,  -86,  -85,  -84,  -83,  -82,
        -81,  -80,  -79,  -78,  -77,  -76,  -75,  -74,  -73,  -72,  -71,
        -70,  -69,  -68,  -67,  -66,  -65,  -64,  -63,  -62,  -61,  -60,
        -59,  -58,  -57,  -56,  -55,  -54,  -53,  -52,  -51,  -50,  -49,
        -48,  -47,  -46,  -45,  -44,  -43,  -42,  -41,  -40,  -39,  -38,
        -37,  -36,  -35,  -34,  -33,  -32,  -31,  -30,  -29,  -28,  -27,
        -26,  -25,  -24,  -23,  -22,  -21,  -20,  -19,  -18,  -17,  -16,
        -15,  -14,  -13,  -12,  -11,  -10,   -9,   -8,   -7,   -6,   -5,
         -4,   -3,   -2,   -1,    0,    1,    2,    3,    4,    5,    6,
          7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,
         18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,
         29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,
         40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,
         51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,
         62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,
         73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,
         84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,
         95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,
        106,  107,  108,  109,  110,  111,  112,  113,  114,  115,  116,
        117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,
        128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,
        139,  140,  141,  142,  143,  144,  145,  146,  147,  148,  149,
        150,  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,
        161,  162,  163,  164,  165,  166,  167,  168,  169,  170,  171,
        172,  173,  174,  175,  176,  177,  178,  179]),
            'lat': np.array([
       -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78,
       -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65,
       -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52,
       -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39,
       -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26,
       -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13,
       -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,
         1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90]),
        },
        # GISS ModelE (Miller et al., 2014)
        '2x2.5_deg_centre_GISS': {
            'lon': np.array([
       -177.5, -175. , -172.5, -170. , -167.5, -165. , -162.5, -160. ,
       -157.5, -155. , -152.5, -150. , -147.5, -145. , -142.5, -140. ,
       -137.5, -135. , -132.5, -130. , -127.5, -125. , -122.5, -120. ,
       -117.5, -115. , -112.5, -110. , -107.5, -105. , -102.5, -100. ,
        -97.5,  -95. ,  -92.5,  -90. ,  -87.5,  -85. ,  -82.5,  -80. ,
        -77.5,  -75. ,  -72.5,  -70. ,  -67.5,  -65. ,  -62.5,  -60. ,
        -57.5,  -55. ,  -52.5,  -50. ,  -47.5,  -45. ,  -42.5,  -40. ,
        -37.5,  -35. ,  -32.5,  -30. ,  -27.5,  -25. ,  -22.5,  -20. ,
        -17.5,  -15. ,  -12.5,  -10. ,   -7.5,   -5. ,   -2.5,    0. ,
          2.5,    5. ,    7.5,   10. ,   12.5,   15. ,   17.5,   20. ,
         22.5,   25. ,   27.5,   30. ,   32.5,   35. ,   37.5,   40. ,
         42.5,   45. ,   47.5,   50. ,   52.5,   55. ,   57.5,   60. ,
         62.5,   65. ,   67.5,   70. ,   72.5,   75. ,   77.5,   80. ,
         82.5,   85. ,   87.5,   90. ,   92.5,   95. ,   97.5,  100. ,
        102.5,  105. ,  107.5,  110. ,  112.5,  115. ,  117.5,  120. ,
        122.5,  125. ,  127.5,  130. ,  132.5,  135. ,  137.5,  140. ,
        142.5,  145. ,  147.5,  150. ,  152.5,  155. ,  157.5,  160. ,
        162.5,  165. ,  167.5,  170. ,  172.5,  175. ,  177.5,  180. ]),
            'lat': np.array([
       -89, -87, -85, -83, -81, -79, -77, -75, -73, -71, -69, -67, -65,
       -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39,
       -37, -35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13,
       -11,  -9,  -7,  -5,  -3,  -1,   1,   3,   5,   7,   9,  11,  13,
        15,  17,  19,  21,  23,  25,  27,  29,  31,  33,  35,  37,  39,
        41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,  65,
        67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89])
        },
        # ACCMIP (Lamarque et al., 2013)
        '2x2_deg_centre_ACCMIP': {
            'lon': np.array([
       -179, -177, -175, -173, -171, -169, -167, -165, -163, -161, -159,
       -157, -155, -153, -151, -149, -147, -145, -143, -141, -139, -137,
       -135, -133, -131, -129, -127, -125, -123, -121, -119, -117, -115,
       -113, -111, -109, -107, -105, -103, -101,  -99,  -97,  -95,  -93,
        -91,  -89,  -87,  -85,  -83,  -81,  -79,  -77,  -75,  -73,  -71,
        -69,  -67,  -65,  -63,  -61,  -59,  -57,  -55,  -53,  -51,  -49,
        -47,  -45,  -43,  -41,  -39,  -37,  -35,  -33,  -31,  -29,  -27,
        -25,  -23,  -21,  -19,  -17,  -15,  -13,  -11,   -9,   -7,   -5,
         -3,   -1,    1,    3,    5,    7,    9,   11,   13,   15,   17,
         19,   21,   23,   25,   27,   29,   31,   33,   35,   37,   39,
         41,   43,   45,   47,   49,   51,   53,   55,   57,   59,   61,
         63,   65,   67,   69,   71,   73,   75,   77,   79,   81,   83,
         85,   87,   89,   91,   93,   95,   97,   99,  101,  103,  105,
        107,  109,  111,  113,  115,  117,  119,  121,  123,  125,  127,
        129,  131,  133,  135,  137,  139,  141,  143,  145,  147,  149,
        151,  153,  155,  157,  159,  161,  163,  165,  167,  169,  171,
        173,  175,  177,  179]),
            'lat': np.array([
       -89, -87, -85, -83, -81, -79, -77, -75, -73, -71, -69, -67, -65,
       -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39,
       -37, -35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13,
       -11,  -9,  -7,  -5,  -3,  -1,   1,   3,   5,   7,   9,  11,  13,
        15,  17,  19,  21,  23,  25,  27,  29,  31,  33,  35,  37,  39,
        41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,  65,
        67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89])
        },
        # GEOSChem (Bey et al., 2001) - 4◦ x5◦
        '2x2.5_deg_centre_GEOSChem': {
            'lon': np.array([
       -180. , -177.5, -175. , -172.5, -170. , -167.5, -165. , -162.5,
       -160. , -157.5, -155. , -152.5, -150. , -147.5, -145. , -142.5,
       -140. , -137.5, -135. , -132.5, -130. , -127.5, -125. , -122.5,
       -120. , -117.5, -115. , -112.5, -110. , -107.5, -105. , -102.5,
       -100. ,  -97.5,  -95. ,  -92.5,  -90. ,  -87.5,  -85. ,  -82.5,
        -80. ,  -77.5,  -75. ,  -72.5,  -70. ,  -67.5,  -65. ,  -62.5,
        -60. ,  -57.5,  -55. ,  -52.5,  -50. ,  -47.5,  -45. ,  -42.5,
        -40. ,  -37.5,  -35. ,  -32.5,  -30. ,  -27.5,  -25. ,  -22.5,
        -20. ,  -17.5,  -15. ,  -12.5,  -10. ,   -7.5,   -5. ,   -2.5,
          0. ,    2.5,    5. ,    7.5,   10. ,   12.5,   15. ,   17.5,
         20. ,   22.5,   25. ,   27.5,   30. ,   32.5,   35. ,   37.5,
         40. ,   42.5,   45. ,   47.5,   50. ,   52.5,   55. ,   57.5,
         60. ,   62.5,   65. ,   67.5,   70. ,   72.5,   75. ,   77.5,
         80. ,   82.5,   85. ,   87.5,   90. ,   92.5,   95. ,   97.5,
        100. ,  102.5,  105. ,  107.5,  110. ,  112.5,  115. ,  117.5,
        120. ,  122.5,  125. ,  127.5,  130. ,  132.5,  135. ,  137.5,
        140. ,  142.5,  145. ,  147.5,  150. ,  152.5,  155. ,  157.5,
        160. ,  162.5,  165. ,  167.5,  170. ,  172.5,  175. ,  177.5]),
            'lat': np.array([
       -89.5, -88. , -86. , -84. , -82. , -80. , -78. , -76. , -74. ,
       -72. , -70. , -68. , -66. , -64. , -62. , -60. , -58. , -56. ,
       -54. , -52. , -50. , -48. , -46. , -44. , -42. , -40. , -38. ,
       -36. , -34. , -32. , -30. , -28. , -26. , -24. , -22. , -20. ,
       -18. , -16. , -14. , -12. , -10. ,  -8. ,  -6. ,  -4. ,  -2. ,
         0. ,   2. ,   4. ,   6. ,   8. ,  10. ,  12. ,  14. ,  16. ,
        18. ,  20. ,  22. ,  24. ,  26. ,  28. ,  30. ,  32. ,  34. ,
        36. ,  38. ,  40. ,  42. ,  44. ,  46. ,  48. ,  50. ,  52. ,
        54. ,  56. ,  58. ,  60. ,  62. ,  64. ,  66. ,  68. ,  70. ,
        72. ,  74. ,  76. ,  78. ,  80. ,  82. ,  84. ,  86. ,  88. ,
        89.5])
        },
        # UKCA (O’Connor et al., 2014)
        '2x3.75_deg_centre_UKCA': {
            'lon': np.array([
       -178.125, -174.375, -170.625, -166.875, -163.125, -159.375,
       -155.625, -151.875, -148.125, -144.375, -140.625, -136.875,
       -133.125, -129.375, -125.625, -121.875, -118.125, -114.375,
       -110.625, -106.875, -103.125,  -99.375,  -95.625,  -91.875,
        -88.125,  -84.375,  -80.625,  -76.875,  -73.125,  -69.375,
        -65.625,  -61.875,  -58.125,  -54.375,  -50.625,  -46.875,
        -43.125,  -39.375,  -35.625,  -31.875,  -28.125,  -24.375,
        -20.625,  -16.875,  -13.125,   -9.375,   -5.625,   -1.875,
          1.875,    5.625,    9.375,   13.125,   16.875,   20.625,
         24.375,   28.125,   31.875,   35.625,   39.375,   43.125,
         46.875,   50.625,   54.375,   58.125,   61.875,   65.625,
         69.375,   73.125,   76.875,   80.625,   84.375,   88.125,
         91.875,   95.625,   99.375,  103.125,  106.875,  110.625,
        114.375,  118.125,  121.875,  125.625,  129.375,  133.125,
        136.875,  140.625,  144.375,  148.125,  151.875,  155.625,
        159.375,  163.125,  166.875,  170.625,  174.375,  178.125]),
            'lat': np.array([
       -88.75, -86.25, -83.75, -81.25, -78.75, -76.25, -73.75, -71.25,
       -68.75, -66.25, -63.75, -61.25, -58.75, -56.25, -53.75, -51.25,
       -48.75, -46.25, -43.75, -41.25, -38.75, -36.25, -33.75, -31.25,
       -28.75, -26.25, -23.75, -21.25, -18.75, -16.25, -13.75, -11.25,
        -8.75,  -6.25,  -3.75,  -1.25,   1.25,   3.75,   6.25,   8.75,
        11.25,  13.75,  16.25,  18.75,  21.25,  23.75,  26.25,  28.75,
        31.25,  33.75,  36.25,  38.75,  41.25,  43.75,  46.25,  48.75,
        51.25,  53.75,  56.25,  58.75,  61.25,  63.75,  66.25,  68.75,
        71.25,  73.75,  76.25,  78.75,  81.25,  83.75,  86.25,  88.75])
        },
        # GEOSChem (Bey et al., 2001) - 4◦ x5◦
        '4x5_deg_centre_GEOSChem': {
            'lon': np.array([
        -180, -175, -170, -165, -160, -155, -150, -145, -140, -135, -130,
       -125, -120, -115, -110, -105, -100,  -95,  -90,  -85,  -80,  -75,
        -70,  -65,  -60,  -55,  -50,  -45,  -40,  -35,  -30,  -25,  -20,
        -15,  -10,   -5,    0,    5,   10,   15,   20,   25,   30,   35,
         40,   45,   50,   55,   60,   65,   70,   75,   80,   85,   90,
         95,  100,  105,  110,  115,  120,  125,  130,  135,  140,  145,
        150,  155,  160,  165,  170,  175]),
            'lat': np.array([
        -89, -86, -82, -78, -74, -70, -66, -62, -58, -54, -50, -46, -42,
       -38, -34, -30, -26, -22, -18, -14, -10,  -6,  -2,   2,   6,  10,
        14,  18,  22,  26,  30,  34,  38,  42,  46,  50,  54,  58,  62,
        66,  70,  74,  78,  82,  86,  89])
        },
        }

    else:
        print( 'No selection of grid version given' )
        sys.exit()

    if just_1x1_grids:
        grids2use = [i for i in d.keys() if '1x1' in i]
        d_orig = d.copy()
        d = {}
        for grid in grids2use:
            d[grid] = d_orig[grid]
        del d_orig
        return d

    else:
        return d

