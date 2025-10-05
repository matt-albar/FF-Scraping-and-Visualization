
# Paste your cookie string here
cookie_string = "gig_bootstrap_3_WPLZFkJ278FjSau2FfLCrTRUyksAjeYpcva-qIfGl71F4VWrI-7Xb5y0snqKXDva=auth-id_ver4; nflcs.prod.crossDomainStorageCleared=true; AMCV_F75C3025512D2C1D0A490D44%40AdobeOrg=179643557%7CMCIDTS%7C20360%7CMCMID%7C11690837921959008619222919677858368451%7CMCAID%7CNONE%7CMCOPTOUT-1759072434s%7CNONE%7CvVersion%7C5.5.0; gig_bootstrap_3_3g_DApOD0TCeN6ZJpzQMr7H1cIbtqtHwDjKVESN3N5oohMleIozT0I9WecPZeytT=auth-id_ver4; glt_3_3g_DApOD0TCeN6ZJpzQMr7H1cIbtqtHwDjKVESN3N5oohMleIozT0I9WecPZeytT=st2.s.AtLtNJ7t_Q.P_3fkI5W-…2footerSeen%22:false%2C%22lastRecPage%22:%22nfl.com:plus:learn-more:landing%22%2C%22percentPageViewed%22:31%2C%22initialPageViewed%22:24%2C%22maxPageViewed%22:1851}}; AMCVS_F75C3025512D2C1D0A490D44%40AdobeOrg=1; ff=uid%3De3f27dabb2c98bee97ccbda38ac429ce%26fu%3D3182142%26tz%3DUS%2FEastern%26env%3Dproduction%26g%3D102025%7CL%403329214~West%20Milford%20Fantasy~4~Luka%20Puka%20Egbuka%26d%3D1757786656%26h%3D63e6d1476e46696f576a17ade9cabd76; at_check=true; mbox=session#e67952664fcb494d8cc3e05e053623f9#1759672389"

# Dict to store cookies
cookies = {}

# Loop through each cookie
for cookie in cookie_string.split('; '):
    # Split cookie into name and value
    cookie_parts = cookie.split('=')
    cookies[cookie_parts[0]] = cookie_parts[1]