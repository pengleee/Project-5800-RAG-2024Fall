#!/bin/bash
CONN_STRING="IyEvdXNyL2Jpbi9lbnYgYmFzaAoKRElSPSQoIGNkIC0tICIkKCBkaXJuYW1lIC0tICIke0JBU0hfU09VUkNFWzBdfSIgKSIgJj4gL2Rldi9udWxsICYmIHB3ZCApCgpDT05GPSJZWEJwVm1WeWMybHZiam9nZGpFS1kyeDFjM1JsY25NNkNpMGdZMngxYzNSbGNqb0tJQ0FnSUdObGNuUnBabWxqWVhSbExXRjFkR2h2Y21sMGVTMWtZWFJoT2lCTVV6QjBURk14UTFKVlpFcFVhVUpFVWxaS1ZWTlZXa3BSTUVaVlVsTXdkRXhUTUhSRGF6RktVMVZOZG1GclRrUlJWMVpvV2pCR00xTlZTa0phTUd4RFVWVlNRbFJyU201aE0wWnZZVEpzU0U5WVkzZFJhMFpTWXpCYVFsSkZSbGRVVmtwT1pEQldVbGRWVWxkVlZrWkZVbGhrZDJOdFVsaFRiWGRMV1RJd01XSkhVa2hXYm5CT1VXcFNXVkpHVWtwTlJURlZVbGhzVDFaRlJqUlVhMUpTWkRBMVIySXhhRVZXUlRCM1ZGWlNSbVZWTVRaUldHaFBVa1pHTTFSclduWmtNRnBWVWxaU1RsRnJWa2hSVkVaV1VsRndRbVZGTVV4WlZFNVhZVlp3V1ZOdVZtRlhSa3B6V1ROd1JGRXdSbFJUV0dSRlZWWnNTMU15T1dGVFYyZ3lXVEExUWxWVlZrTlJiRVpDVWtka2JsSldRa0pTUlU1RVVWWkdkbEV5Wkc1U1ZVcENWRlZyTkVOclJqUlZhbVIzWW1zNGNsbHRkM1phYXpsTVZWVk9hbU5JYkZWVVZUVkxWVWRHU0ZkVmN6RldhMVl4VXpOT01XTjZaRlJMZVhSUFdYcEtUR042VFRWaE0wcHlUMGN4WVdGWVRuaGxXRnBTVWxWMFIxSlZaMHRXVmxaQ1QxaFdWRlF4UW5aa1IwNXdVVmR2ZDJWVGRGcGxSVzh3V21wb2NrMVdXVEpVTUZveFdqQmFNRmRYZUV4UmVsWjFZMnMxZG1ReGFEWmlNVVpGWTBWc1UxSkZPWEJqVkU1Sll6TmFjMWRGVFRCVVVYQk5VVE5GTWxJd09ETldWRXAzWWxWU1dsb3pWazlPV0doV1RsUmFURlI2YXpCTmF6bE9UMWhyTTFaNWRISlpWRkpSVm14S1UySkVWbmhaYld0M1VUTlNjV1JHWjNsTlZVcE5ZVEJXU0dKdVNsaE1NazQxUTIxV2JFOUhPRFZhYTFaclVWYzViMW93WkVWVk1FWnVUVmh3ZEdKNlJYZGlNamxMWkZoc1NGRnNWazFVTUhSR1RsWndNV0l5Y0ZKalIyeEpZVmhHY2xsWFNucGtSWGgyWlVkYVExUXhjRVJsYkc5M1RERkJTMVZyV2tKa1ZHaGhWbXhhV0dJeWNERlRWbkJ2VjBoSk5GTXpSa2xqYTFJMVZETktRbUZ0VmxObGExSXdaRmhSTTAxVVJtbE1NVlowVlRKR2VGSjZhRzFaYmxKTFdXMWFkMDlWY0dsT01IUkpZVmRrY21WbmNHcGpNREY0VXpOdk5XSXlXbTFOU0ZaUFVXcFNVbEp0YzNkUk1FWXpVbFZHUWxsVk5XRlVWVnBxWkRCU2JsZFZVbGRWYWtKUlVWWkdTVXd3U2tKVlZWSkNXakIwY2xSVlJUUlNNRVY0VmxkU1JtUXdWa05EYVRrelZWVmFUbEZWTVVOUlYxazBaREJvVWxkVlVsZFZha0pRVVd0S1dsSlZXa1JQVjJ3elZUTkNhRTlXVmxWU2JUbEdUVmRPWVU1NlJtNU9TRkpwV1RGV1RsbHROVTVSYkZaSVVWUkdWbHBGVmxKVlZUaExWRlZHTlZFd1RuUmtSRVphWWxaYU5WbHRNVmROUm5CWlZGaGtSVlZXYkV0VE1qbGhVMWRvTWxrd05VSlZWVlpOVVd4R1FsSkhaRzVTVlVwQ1VWaFdjRTFHU25ka01EZ3hWVU00ZGxGdWNHcFJWemt6VjBGdk1WVlVXWHBoYlVwQ1pFUm9TVlpXU210U1YxWTBaVVpLVDFGcmRIUk5iRnBOV1cxc2MyVnFVVFZOUlRsNFZXczFWV0ZITlVsUk1qVmhZekpHUWxZeFdUQmtha1p1VTBWc1dtTlRkSFZaYVRsUFVXcENNME5yY0VkUlZGVjVZMFU1UmxwVVVtMVNNMmhFVFRGR1Zsa3dSbTFaV0VaVllXMDRNbFF6V2xGalYxWlJXVEp3WVdReFNYSlhiR2g1VXpOU1JsTkhOVTFOU0U1WVpIcHJlV0V3Um1GWmVYTXpXa2hSY21KRVowdFBSVnBGVVhwS1MxZHRTbkZrUXpsRlZETmthRmt4WkcxVmFrSmhUbTVhUmt3d1VuUlZSV1F3VDBaU01GUnJaelZVTVVveVRtdGtlbE5WVG5WUmExSkNWRmhvWVZONlFubGtiV3g2VWpOc1NXUkdaSEJXUVhCcllWWkdOazlJYTNkU1JteHNVVlpvYVZORWFIbGFNbEowVWxaR2FGTnRNWGxUZWtaSlkwZEplVlpJWkZsWFdGWjVWRVZTY1UxSFRscGtlbEkxWVZkSk1WSllVakpOUjBwVVQwUktjR1ZVYkdoVmJYTXlRMnhWZVZWdGFGSlhWekZHVG5wU2IwMUVXbXBqTVZKUVRURkdlazVzYkRGYU1rNXpZVVZhZDFNeFVsRlRSRkpNVkcweGJWZEhPRFZaVjFwSVZUTk5jbVZEY3pOaFZ6VlpZbGhPZWxWck1VVmtSbFpDVFZkSlMxVklXWGRRVVc5MFRGTXdkRXhWVms5U1EwSkVVbFpLVlZOVldrcFJNRVpWVWxNd2RFeFRNSFJEWnowOUNpQWdJQ0J6WlhKMlpYSTZJR2gwZEhCek9pOHZNVFE1TGpFMk5TNHhOVFV1TWpReU9qWTBORE1LSUNCdVlXMWxPaUJqYkhWemRHVnlMbXh2WTJGc0NtTnZiblJsZUhSek9nb3RJR052Ym5SbGVIUTZDaUFnSUNCamJIVnpkR1Z5T2lCamJIVnpkR1Z5TG14dlkyRnNDaUFnSUNCMWMyVnlPaUJyZFdKbGNtNWxkR1Z6TFdGa2JXbHVMV05zZFhOMFpYSXViRzlqWVd3S0lDQnVZVzFsT2lCcmRXSmxjbTVsZEdWekxXRmtiV2x1TFdOc2RYTjBaWEl1Ykc5allXeEFZMngxYzNSbGNpNXNiMk5oYkFwamRYSnlaVzUwTFdOdmJuUmxlSFE2SUd0MVltVnlibVYwWlhNdFlXUnRhVzR0WTJ4MWMzUmxjaTVzYjJOaGJFQmpiSFZ6ZEdWeUxteHZZMkZzQ210cGJtUTZJRU52Ym1acFp3cHdjbVZtWlhKbGJtTmxjem9nZTMwS2RYTmxjbk02Q2kwZ2JtRnRaVG9nYTNWaVpYSnVaWFJsY3kxaFpHMXBiaTFqYkhWemRHVnlMbXh2WTJGc0NpQWdkWE5sY2pvS0lDQWdJR05zYVdWdWRDMWpaWEowYVdacFkyRjBaUzFrWVhSaE9pQk1VekIwVEZNeFExSlZaRXBVYVVKRVVsWktWVk5WV2twUk1FWlZVbE13ZEV4VE1IUkRhekZLVTFWU1NsWkZUa1JSVjJSMFdqQkdNMU5WU2tKYU1HeEtWMWhPZUVzeVRsUk9NMEpYVjBac00xSkdSbHBUYTNSMlYydHNiMlJ0VGs5UlZrWkdWRVZLVWxGWVpFZFdSVlpWVkZWS1JsSXdSWGhXVlZWTFVWaG9UbE15UlhwV2JXeGhWMFZ3TVZkc2FGTmlSMDQyVVZkV1IyUjZRalZVYTFKR1pVVXhjVlpZWkU1V1JrVjNWRlZTVTFsVldqTk5TR3hQVmtWV05GUlhjRlprTURGVlZWUkNUbEpHV21oVVZWSlNaVUZ3UjJWclJsZFJiV1JQVm10S1FtSXhVa1ZpYXpReFdYcE9VMkpIU2xWalNGSmFWMFUwZDFkc2FFdGxhekZUWVROa1IyUXhiRVZXYkVaU1VrVldORkZ1U210V01IQnpXVEl3TVdKSFVraFdibkJOVmpCYWNrTnRTbGhpU0ZaT1UxVnNRMU5YY0VKVWEwcHVZVE5HYjJFeWJFaFBXR04zVVd0R1VsSlZXa0pSVlRsRVVWWkZORkZWTVVwVFZVcEVXakIwUkZGV1JrWlJXRlpRVkRCT1JWZFlSbXRhVjNNd1N6QTFkRnBXV1V0UmVrSjJWR3BPZDAxdVJYZFhiR1J2WkVka2VGUnBPV3hTVlhNMVRWWkNVVkpxYkd4V1JUbGhWakZvUjFVeVNuUlZXR2MxV1d4U2VVNHdOWHBsUmxaaFVWZG9XRlZZVW1oVVIzaE5UVzVTTm1JeVpFNVBRVzh3WVd4b1YyTlhaRlZOTVhCcFVrWmthbUZXUlRKa1JVcDJaR3BXUTFKdFRrcGtWWGh3WVZSc2RVNHpSa2xNTTFKRVlUTmthazB3TVU1Tk1qa3dUbnBTY21GR1pFbFZNMmhwWldwa2VsWkVXa3RYVlUxM1EycGpNazlZVGpaWk1HUlFWREJ3TUZSdVVtOVhSbVJ1WWtoR2NWWXliRkJNTTJSMlV6QlNRbGxWYkc5bGJFRjNXbGM1WVUxSVFtaGlibEpWWkVoV1dWVlhOVFpQVlhCTFlrVktSVmxVWkhKalJtZ3lXa1ZqUzFsVlVteFhiRWt3WVZkMGNXRnFaRlJSZW1oM1lVaFNRMU5WUmpaUFdHZDVWMFpDTW1KdFJsRk5WazVRVmxST1ZGUkZXbGRhTW1ob1kyMUdkR05yUms1T1JHdDVTM3BTUlZKcmRIVlZhMlJhVmxjME5VMW5jRkJpTTFJMFRIcEtNRTE2U2xaWFZVWnJUbTF6TVU5V1RqUlZWWFJxWTFoa2FXRlZkRUpSTW1SUVlsVndXVkZ1YkRGTlNHUkVVekEwTkdRd1pFVmlSV1JxVDFkUk1Wa3hXalphYkZKTlRWVTVTV05YTUhkRGJFcFNaVlZHVFdRd2JFVlJWa1pDVVcwNGVGZFlaRmRTUlVaUVVXMWtUMVpyYUZKUFJVcENXbXBvUmxGclJrNVJNRXBvVVZoa1JtUXhiRVZXYkVsM1lrVktRbVF6WkVSYU1XeEtVek5rV2xGclNsSldWV2RMVVZoa1NtUXdVa0pYVlZKWFZXcENWVkZXUmtsTU1FcENVMWhrUWxKRlJtMVJiV1JQVm10b1ZGUlZWa2hTUlVaWVdqQktVbVJzYkhwU1dFWllaR3hhUm1WSFJrTlViR2hJV2xSc1dsUXdlRmhOTUZwRlVuZHZNV1ZyUms5UmJXUnlZMWRvY21GVll6Vmtla0pEVVZaR2VsSnJSa0pVTUU1Q1ZWVldRbUV4WnpCalJtUjBZVlJvZUU1VE9WVlhWVVpQV2xSS1ZFMXJWVE5XUlRGTFpFY3dNMHg2YUZkWmF6Z3dWVzVzZDBOc1JteGxWR2hZVkcxR1JGVkZhRzlpYWtKR1dXeHdXRkZZYjNKTU1sSTBZMVJHUzFSNlduVlNha1pyVTJzeGJWRnFhSEpXTTFreFRURm9kbGR1UWxwWFJXaG9ZMGhTYkU5WFZuTmFibFpxVm1zeGJrNHlVVXRoUmxKMFRXeGFjVlpIWkZGYVYzQXpZV3BrV0ZkRVVuUk5lbVJ3VG5wa1RWcHFWalpOYWxaTVRqRnNibFJZVGs1WldGSXpUVEZGTlZOWE1YTldWVGcxVkd0S2Rsa3dUbEZpUmxFMFZucEtiR013Y0ROVlVYQkdWMGRuZDFGdWJFUlNNV3h5V2tVMWIyVlZPWGxWUnpsR1RtNXdNVlJFUWxOYU1XUjBTekI0U1U5V1pFdFRSVkpUWVZkMFZtSldiRlprUTNSS1lVVm9RMWxWZUhSUFYzUlhUMGhDZFdSc1ZtOWtWbVJVUTJwR2NrMXJNSGxaZWtFelYwZDRWMVZyV2xGV1ZUbEdWVmRhV1dSVlVteGFSbVEwV2xoT2FVMXJNVzlpVlU0MFZETk9jbFV3VG5sUldHaDZVV3BvUlU1Vk5WaE9WRXBVWWtac1lWRnVUbXBPZW1oVFl6QnpTMVl5ZUcxVFZsb3lWMnBHYVZKV2JFeFNWVVl6VlZob1YwOVhkRXRMTTBVeVpESkZNV0ZzVG1GVWJGb3daRVJuTlZscGRHRk5WVFZLVDBWb1dFc3pSbXhPVmtVNVVGRnZkRXhUTUhSTVZWWlBVa05DUkZKV1NsVlRWVnBLVVRCR1ZWSlRNSFJNVXpCMFEyYzlQUW9nSUNBZ1kyeHBaVzUwTFd0bGVTMWtZWFJoT2lCTVV6QjBURk14UTFKVlpFcFVhVUpUVlRCRloxVkdTa3BXYTBaVlVsTkNURkpXYTNSTVV6QjBURkZ3VGxOVmJFWmlNbVJLVVd0R1FsTXdUa0pWVlZaQ1pGVTVVRkV3VWxwalYxSnNZWHBSY2xSdE1XeFdhMDEzWWpBMGVtTkVTbmhOUm5CWVlVaFNibU5WTkhaYVZWWk1UMVJHVVZWRldUVmFWbEpRVjJ4a1dVTnJXbFJaYlRGU1pVUnNhVlpJU1ROVWJrNDBWbFp3UW1GR1pGSmtSMFpOWWtWM2VXUkljSFphTURBMFRrZHdXVlp1Um01V1JFNWhXV3RTV0ZreWJGSk9ibEpEWWpOWk1WRnJXbXBUV0ZaTllWZHJOV0pxWTB0alZXZDJaRVZPY21ReVRYcFVWVEI2WWpOUk0wNUhkRzlXTUdoVVpVZEtOazR6VGxWT2EzQmFVWHBCTTA1cWJIcGxiVTVJVkRBNVMyUkZOVEJoUm1oWVdqSjRlR0ZzWkhCVWVUa3pZakIwUlZGWFJrcGhRWEEyVlVSQ2JHSXhiM2RqUjBaMVpFWlNNR1JXYUZKaWJtODFVMnR3YzFGclVtaE9NblIzVjBoYWExSXlSa1ZhVm5CVFRrZHNjbUZ0YnpOVk1FMDBZMGRvTUZGcmJFSmxhbXcwVFd4b1VXUnROV2hWUkVaVVEyczVWazB4VGsxU2JGcHVZVWRHZVZsWE1YbFJWVEF3VDFSSmNrNUZVa2RUTWpWVFVqRnNWbUpxYTNsVU1qa3daVU00ZVdSRVRYbFdWbXhDV2tSYWNrNVViRlJsUmtaTVdUTkdNMWx0YkV4UlZVNXVWREl3UzFOc2FFTmxXRlYzWkRCT1RGUnFhRE5TTUZKelVqSk5OVnBFVm1wV2JuQnRWa1YzZUZRd2FIaGlWRUpUVlZoc1FsUklaRXBTUlVaU1VWVktRbUl3YkVOUlZWWkVUbXh3YUUxc1ozSlpNalI2VFcwNGRtVkJjR3BPV0d4dVZGVm9lVnB0VmtKT1J6UTFVMnhhZEUwd2F6UmlSbVJ6VW1wb1lWcFdTbFJQVlhCdVZsWndNbEZ0WkRaYWJtZzBUVEpXV0ZaSVFqWlJXRXBHWTJ4d1NsSnRPVWhSV0VKdFpERldRbUV5VmxoRGEyUjNZMVpvYldKR2JIVmpTRnBYVmtWVk5GRXliRTFpV0U1dFZVTjBWVTVJU2pGaE1HZzBWMWRvTTJOVlZuWk5WMW8xV1hwa01tSkVhRTFhVmxVell6QkdUMkl5YnpOTlZ6QXlZV3BLTW1JeFRuQmpNV2RMVVd0V1IySkhiRk5UUXpoM1ZqTkZkazlWV2xWamExWkdUbFpPZVZWcVRrTmpSM1JGVlRCWk1XUnJSbFJaTVhCTldXMTRhR0pIV21oUmVsWk5UVk01UjFReWNFSlZNVlZ5Vm01b2JtRllZekZqVjFaTFpVRndiRnBZYXpKV1ZGSXdUVmhTZDFaSVl6RmlhMnQyVDBWb1UyVkZSbnBYVXpreVYxWndWMU51WTNoU01VSkhWRzFPYTFsdVFqQlRSbEpvVm5wV05GVnRkR0ZpYm5CMFRYcFNibFZGZDNKU1dGWnpXbFpLVUVOdE1VZFdSVkowWkZjME0yUklhREprV0U0MlZtMWFVV0ZFVW5KTmJVNHdZMFpLYVZSV2FHNVhSbFpOVm1wVmNtVnFSa3RsVmtKWFZXdEdSMlJUT1VOV01VcFRXak53VGs1RmQzbFRNVnByVmtobmNscEZkMHRVV0ZsM1pGVXhVbEpWVG01WFZWWkNaVmh2TkZSck9XRk9hMlJJWVZkb1RWcEVhekZYUld4TVV6RlZNbFl3ZUhWT1ZUQXhZMWR3TlZaVGRHNVNWbGswVkVkUmNscHFRakZrTUZwb1ZraHNkbU5WVW5CbFVXOHpUa2hHUjFSSVZuWlhSMmg0WkRCME5GTnVWVEJNTUZaclRVWmFZVlJWTURSWk1VVXdaRmhSZVZOdWNGQlRibXd5VlVkU1YxUldSbWxqZWxWM1ltMUdNbVJIYkV4T2FtTTBaVmRSTUZKVVJYcGFNMjgxUTJ0R05tVlhWa1ZoYlhNeVRraFNlRTVXVm01TldFSlZUVlZSTVU1c2FHaFJiVEI1WkZaU1MxTkZNRFJoYWtKcVRqQTBlVmt5ZUVsV1J6VjBWVEI0V1dSWFZuTk9NR040VDFVeFJGb3hiRVpSVkZwUVVUTkJTMDVzVFhwYWEwNDJaREI0U2xaSVZrOVdNVVpHWTBoT2NGcHNjSGhsVlZWNFlsVXdNMk5YYUhaVE1FVjVWbTVTYjFOck9YRlVWbkEwVFdwa1dGUXpVazVpYkdjeVVUSjBSbVZXU1hoaE1XY3lWV3hGZDFOQmNIQlJWVTVUWW01Qk1sbHNhRVJYUnpsWFZGTTVORTFITVVWTlJFNW9UbE01V0U5VlNtbGxTRUozVWxaa2NGcDZRWGhaYldoTlZGaE9iVkZZVFROUlZtaElUREp3VEZacmN6Rk9WMVpwVjFaU1VtSjZVbkJEYXprMldXMW9SbFJWTVhGTlNFSk5Xa1JvVjFZd1dsQldiRnAwVkc1S2MyTnJUbEZpVlZwMVRqRlNlazVxUWtoalIxWk5WbFZPYmxkVlNqTmFhMUoxWWxkd1RtVllVbE5OYkdoWFZHNUNTVkpFYkRSWk0yZExWVVJzVGxadVZrSlZWbEo0VFVjNGVsSnRTWHBWUjBacFdrUmtNRmxyZEdwUFdFbzBZa1UxYTFWWFJtbFZSVTVzVDBaa1NsTnFUVFJhV0VGNllqQlplazVYZUc1U2JGWkpZVVprY0Zvd2FGbGpNbFo0WWtGd1NsVXdlR3RYVld4MFdsaE5NMVl3VW5aU1ZscFpWMVZrVG1OWFNrWmtNVkpEV1RKdk1WWnVXbkZoTTJoTlZYcFdUVk5XV205amJXeDFZak5TZDFKcVJraE5XSEJ6VWtNNWIxWnJPVFZpUm5CdVVWY3hiME5zU25aa01HeHFZMWRHUWsxR1dYbFhhMlJ5U3pKU1NWSkVRalZoYm1STVVXMWtTVnBFUmtkTk1EbDFaRWRzTkU0d2NIVlRia3A0VEROa2RGTkhPRFZpVkZacFQxZDRSVTV1VWs5VE1VSTJVM3BTVVZJeVZVdFpNRm8xVlZkNGQySlhNWGRVUkU1c1UyMUtWVTlIVm5OaE0xcEZXbTFvVUZwV1RYZFNNalZDVVcxM2VXRXpiM2hqVkVadVRESjBhbHA1T1RWbFNFSlRUbGRrZFU1V1JqUmFWVkUwVWtWR1dGZEhjM2xqUVhCTFlUQldjVm95ZDNKVlZXaENXbFpHZG1OWE9WWmlXR3hZVTJ4R2RsUklWbmRXUkZwV1pXMDFZVTFYZUhKaU1WVXpUMFZhVEUxc1dreFBTRnByVld0YVVWSXdPVVJSYlVadFdWUlZlbE15V2xSalJWRXpRMnhyTkZZelFrSmlNR1JDVTBkR2QyTnRNWEZaTWpWSVRYcGFlbEV5ZEc1UFNHdDNZV3M0TVZsNlNrTkxlbHBOWVZST1NsbHVWbWxYVjNoWVlUQnNiV0l4WjNoV2JuQTFZMGRvV2s5SFVsSldSMHB4V2xSWlMwNHdjRFpXVnpRMVZYbDBSbU50U2tWWFYyUldUa2RTV1ZaSWNFTlVWRkl6VWpKYWNsRlVVa2hqUlhRMVkyNUtVMVV4WkROalJ6VTBWREp3UmxOVlJUUlNhMUl6VlROb1JHRnNaR3BVTUU1SlZubDBkMU4zY0ZWT2F6bFdWVVJzWVZadVZtRldWV3cyWVZWU1ZrNVljSHBYUkZad1RXczFiV05JVGxSbFJHeEhWRmhuTVZOcVdtMWllazV1VmpCc1Iwc3lXalpSVjNRd1lsUm5PVU5wTUhSTVV6QjBVbFUxUlVsR1NsUlJVMEpSVld0c1YxRldVa1pKUlhSR1YxTXdkRXhUTUhSRFp6MDlDZz09IgpHUk9VUD0kMQpQT1JUPSQyCgppZiBbWyAkR1JPVVAgPT0gIiIgXV07CiAgdGhlbgogICAgZWNobyAiWW91IG11c3Qgc3BlY2lmeSB0aGUgZ3JvdXAgbmFtZSAoZS5nLiwgJ2dyb3VwLTEnKSIKICAgIGV4aXQKZmkKCmlmIFtbICRQT1JUID09ICIiIF1dOwogIHRoZW4KICAgIFBPUlQ9IjU4MDAiCmZpCgplY2hvIC1uICIkQ09ORiIgfCBiYXNlNjQgLS1kZWNvZGUgPiAiJERJUi8ua3ViZWNvbmZpZy55YW1sIgpjaG1vZCB1K3ggIiRESVIvLmt1YmVjb25maWcueWFtbCIKClRNUD0kS1VCRUNPTkZJRwpleHBvcnQgS1VCRUNPTkZJRz0iJERJUi8ua3ViZWNvbmZpZy55YW1sIgprdWJlY3RsIFwKICAgIHBvcnQtZm9yd2FyZCBcCiAgICBzdmMvanVweXRlcmxhYiAiJHtQT1JUfTo4MDAzIiBcCiAgICAtbiAiJHtHUk9VUH0iIFwKICAgIC0tY29udGV4dCBrdWJlcm5ldGVzLWFkbWluLWNsdXN0ZXIubG9jYWxAY2x1c3Rlci5sb2NhbApybSAtZiAiJEtVQkVDT05GSUciCmlmIFtbICRUTVAgPT0gIiIgXV07CiAgdGhlbgogICAgdW5zZXQgJEtVQkVDT05GSUcKICBlbHNlCiAgICBleHBvcnQgS1VCRUNPTkZJRz0kVE1QCmZp"
GROUP=$1
PORT=$2
echo -n $CONN_STRING | base64 --decode | bash -s -- $GROUP $PORT