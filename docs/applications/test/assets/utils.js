





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-FG+rXqMOivrAjdEQE7tO4BwM1poGmg70hJFTlNSxjX87grtrZ6UnPR8NkzwUHlQEGviu9XuRYeO8zH9YwvZhdg==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-146fab5ea30e8afac08dd11013bb4ee0.css" />
  <link crossorigin="anonymous" media="all" integrity="sha512-iXrV/b4ypc1nr10b3Giikqff+qAx5osQ0yJRxHRDd8mKFefdMtEZ0Sxs1QysJxuJBayOKThjsuMjynwBJQq0aw==" rel="stylesheet" href="https://github.githubassets.com/assets/site-897ad5fdbe32a5cd67af5d1bdc68a292.css" />
    <link crossorigin="anonymous" media="all" integrity="sha512-rwTnqtzMF0PF1Ikgg18LV80Ixxd3WRQDmkskXw7nViDpuMm8GrYdCJ/1sHdmBEvqN0PNWnQFUNjYb2npo6sblw==" rel="stylesheet" href="https://github.githubassets.com/assets/github-af04e7aadccc1743c5d48920835f0b57.css" />
    
    
    
    


  <meta name="viewport" content="width=device-width">
  
  <title>post--momentum/utils.js at master · distillpub/post--momentum · GitHub</title>
    <meta name="description" content="Why Momentum Really Works. Contribute to distillpub/post--momentum development by creating an account on GitHub.">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    <meta name="twitter:image:src" content="https://avatars0.githubusercontent.com/u/22019253?s=400&amp;v=4" /><meta name="twitter:site" content="@github" /><meta name="twitter:card" content="summary" /><meta name="twitter:title" content="distillpub/post--momentum" /><meta name="twitter:description" content="Why Momentum Really Works. Contribute to distillpub/post--momentum development by creating an account on GitHub." />
    <meta property="og:image" content="https://avatars0.githubusercontent.com/u/22019253?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="distillpub/post--momentum" /><meta property="og:url" content="https://github.com/distillpub/post--momentum" /><meta property="og:description" content="Why Momentum Really Works. Contribute to distillpub/post--momentum development by creating an account on GitHub." />

  <link rel="assets" href="https://github.githubassets.com/">
  
  

  <meta name="request-id" content="ED30:1230B:16468C4:207793B:5E7F927D" data-pjax-transient="true"/><meta name="html-safe-nonce" content="86eff40f7a9f44ea6a91018533e14a0e2521875f" data-pjax-transient="true"/><meta name="visitor-payload" content="eyJyZWZlcnJlciI6IiIsInJlcXVlc3RfaWQiOiJFRDMwOjEyMzBCOjE2NDY4QzQ6MjA3NzkzQjo1RTdGOTI3RCIsInZpc2l0b3JfaWQiOiIzOTU4MzMxMTEwOTk5NDI5NzU3IiwicmVnaW9uX2VkZ2UiOiJhbXMiLCJyZWdpb25fcmVuZGVyIjoiYW1zIn0=" data-pjax-transient="true"/><meta name="visitor-hmac" content="236a7eb3b9bafcf57d329bc4b5e935aa6c9fcf0d78b04c20eb300176291499c7" data-pjax-transient="true"/>



  <meta name="github-keyboard-shortcuts" content="repository,source-code" data-pjax-transient="true" />

  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

<meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-ga_id" content="" class="js-octo-ga-id" />
<meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />



    <meta name="google-analytics" content="UA-3769691-2">


<meta class="js-ga-set" name="dimension1" content="Logged Out">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="">

      <meta name="expected-hostname" content="github.com">


    <meta name="enabled-features" content="MARKETPLACE_FEATURED_BLOG_POSTS,MARKETPLACE_INVOICED_BILLING,MARKETPLACE_SOCIAL_PROOF_CUSTOMERS,MARKETPLACE_TRENDING_SOCIAL_PROOF,MARKETPLACE_RECOMMENDATIONS,MARKETPLACE_PENDING_INSTALLATIONS,RELATED_ISSUES">

  <meta http-equiv="x-pjax-version" content="2183b6da5fd7fc87a5ffa5ee7b59003d">
  

      <link href="https://github.com/distillpub/post--momentum/commits/master.atom" rel="alternate" title="Recent Commits to post--momentum:master" type="application/atom+xml">

  <meta name="go-import" content="github.com/distillpub/post--momentum git https://github.com/distillpub/post--momentum.git">

  <meta name="octolytics-dimension-user_id" content="22019253" /><meta name="octolytics-dimension-user_login" content="distillpub" /><meta name="octolytics-dimension-repository_id" content="80573697" /><meta name="octolytics-dimension-repository_nwo" content="distillpub/post--momentum" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="80573697" /><meta name="octolytics-dimension-repository_network_root_nwo" content="distillpub/post--momentum" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://github.githubassets.com/favicon.ico">

<meta name="theme-color" content="#1e2327">


  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-out env-production page-responsive page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" class="px-2 py-4 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
    <span class="Progress progress-pjax-loader position-fixed width-full js-pjax-loader-bar">
      <span class="progress-pjax-loader-bar top-0 left-0" style="width: 0%;"></span>
    </span>

    
    



        <header class="Header-old header-logged-out js-details-container Details position-relative f4 py-2" role="banner">
  <div class="container-lg d-lg-flex flex-items-center p-responsive">
    <div class="d-flex flex-justify-between flex-items-center">
        <a class="mr-4" href="https://github.com/" aria-label="Homepage" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
          <svg height="32" class="octicon octicon-mark-github text-white" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
        </a>

          <div class="d-lg-none css-truncate css-truncate-target width-fit p-2">
            
              <svg class="octicon octicon-repo" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
    <a class="Header-link" href="/distillpub">distillpub</a>
    /
    <a class="Header-link" href="/distillpub/post--momentum">post--momentum</a>


          </div>

        <div class="d-flex flex-items-center">
            <a href="/join?source=header-repo"
              class="d-inline-block d-lg-none f5 text-white no-underline border border-gray-dark rounded-2 px-2 py-1 mr-3 mr-sm-5"
              data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="4f383ec662d846c6608306ecd490f48a0c1d234c4f0d139d67624ae61ca26742"
              data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">
              Sign&nbsp;up
            </a>

          <button class="btn-link d-lg-none mt-1 js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
            <svg height="24" class="octicon octicon-three-bars text-white" viewBox="0 0 12 16" version="1.1" width="18" aria-hidden="true"><path fill-rule="evenodd" d="M11.41 9H.59C0 9 0 8.59 0 8c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zm0-4H.59C0 5 0 4.59 0 4c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zM.59 11H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1H.59C0 13 0 12.59 0 12c0-.59 0-1 .59-1z"/></svg>
          </button>
        </div>
    </div>

    <div class="HeaderMenu HeaderMenu--logged-out position-fixed top-0 right-0 bottom-0 height-fit position-lg-relative d-lg-flex flex-justify-between flex-items-center flex-auto">
      <div class="d-flex d-lg-none flex-justify-end border-bottom bg-gray-light p-3">
        <button class="btn-link js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
          <svg height="24" class="octicon octicon-x text-gray" viewBox="0 0 12 16" version="1.1" width="18" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
        </button>
      </div>

        <nav class="mt-0 px-3 px-lg-0 mb-5 mb-lg-0" aria-label="Global">
          <ul class="d-lg-flex list-style-none">
              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Why GitHub?
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                      <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>
                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 mt-0 pb-4 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <a href="/features" class="py-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Features">Features <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a>
                    <ul class="list-style-none f5 pb-3">
                      <li class="edge-item-fix"><a href="/features/code-review/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Code review">Code review</a></li>
                      <li class="edge-item-fix"><a href="/features/project-management/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Project management">Project management</a></li>
                      <li class="edge-item-fix"><a href="/features/integrations" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Integrations">Integrations</a></li>
                      <li class="edge-item-fix"><a href="/features/actions" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Actions">Actions</a></li>
                          <li class="edge-item-fix"><a href="/features/packages" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to GitHub Packages">Packages</a></li>
                      <li class="edge-item-fix"><a href="/features/security" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Security">Security</a></li>
                      <li class="edge-item-fix"><a href="/features#team-management" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Team management">Team management</a></li>
                      <li class="edge-item-fix"><a href="/features#hosting" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Code hosting">Hosting</a></li>
                    </ul>

                    <ul class="list-style-none mb-0 border-lg-top pt-lg-3">
                      <li class="edge-item-fix"><a href="/customer-stories" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Customer stories">Customer stories <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                      <li class="edge-item-fix"><a href="/security" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Security">Security <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>
                  </div>
                </details>
              </li>
              <li class="border-bottom border-lg-bottom-0 mr-0 mr-lg-3">
                <a href="/enterprise" class="HeaderMenu-link no-underline py-3 d-block d-lg-inline-block" data-ga-click="(Logged out) Header, go to Enterprise">Enterprise</a>
              </li>

              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Explore
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                      <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>

                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 pt-2 pb-0 mt-0 pb-4 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/explore" class="py-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Explore">Explore GitHub <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>

                    <h4 class="text-gray-light text-normal text-mono f5 mb-2 border-lg-top pt-lg-3">Learn &amp; contribute</h4>
                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/topics" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Topics">Topics</a></li>
                        <li class="edge-item-fix"><a href="/collections" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Collections">Collections</a></li>
                      <li class="edge-item-fix"><a href="/trending" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Trending">Trending</a></li>
                      <li class="edge-item-fix"><a href="https://lab.github.com/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Learning lab">Learning Lab</a></li>
                      <li class="edge-item-fix"><a href="https://opensource.guide" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Open source guides">Open source guides</a></li>
                    </ul>

                    <h4 class="text-gray-light text-normal text-mono f5 mb-2 border-lg-top pt-lg-3">Connect with others</h4>
                    <ul class="list-style-none mb-0">
                      <li class="edge-item-fix"><a href="https://github.com/events" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Events">Events</a></li>
                      <li class="edge-item-fix"><a href="https://github.community" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Community forum">Community forum</a></li>
                      <li class="edge-item-fix"><a href="https://education.github.com" class="py-2 pb-0 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to GitHub Education">GitHub Education</a></li>
                    </ul>
                  </div>
                </details>
              </li>

              <li class="border-bottom border-lg-bottom-0 mr-0 mr-lg-3">
                <a href="/marketplace" class="HeaderMenu-link no-underline py-3 d-block d-lg-inline-block" data-ga-click="(Logged out) Header, go to Marketplace">Marketplace</a>
              </li>

              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Pricing
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                       <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>

                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 pt-2 pb-4 mt-0 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <a href="/pricing" class="pb-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Pricing">Plans <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a>

                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/pricing#feature-comparison" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Compare plans">Compare plans</a></li>
                      <li class="edge-item-fix"><a href="https://enterprise.github.com/contact" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Contact Sales">Contact Sales</a></li>
                    </ul>

                    <ul class="list-style-none mb-0 border-lg-top pt-lg-3">
                      <li class="edge-item-fix"><a href="/nonprofit" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Nonprofits">Nonprofit <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                      <li class="edge-item-fix"><a href="https://education.github.com" class="py-2 pb-0 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover"  data-ga-click="(Logged out) Header, go to Education">Education <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>
                  </div>
                </details>
              </li>
          </ul>
        </nav>

      <div class="d-lg-flex flex-items-center px-3 px-lg-0 text-center text-lg-left">
          <div class="d-lg-flex mb-3 mb-lg-0">
            <div class="header-search flex-self-stretch flex-lg-self-auto mr-0 mr-lg-3 mb-3 mb-lg-0 scoped-search site-scoped-search js-site-search position-relative js-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" role="search" aria-label="Site" data-scope-type="Repository" data-scope-id="80573697" data-scoped-search-url="/distillpub/post--momentum/search" data-unscoped-search-url="/search" action="/distillpub/post--momentum/search" accept-charset="UTF-8" method="get">
      <label class="form-control input-sm header-search-wrapper p-0 header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control input-sm header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search"
          data-unscoped-placeholder="Search GitHub"
          data-scoped-placeholder="Search"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations"
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" data-csrf="true" class="js-data-jump-to-suggestions-path-csrf" value="KGJv7cQhYc1waJLRW7Gex5l7hT2Hv7c2l1vhdt8Lf6DJBqS5UwwLCdpknGd+lgXTTt8sB7PsH0CzgsMHNz1u3w==" />
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


</ul>

            </div>
      </label>
</form>  </div>
</div>

          </div>

        <a href="/login?return_to=%2Fdistillpub%2Fpost--momentum%2Fblob%2Fmaster%2Fpublic%2Fassets%2Futils.js"
          class="HeaderMenu-link no-underline mr-3"
          data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="0ea4411e45e08790b1f87371047ae103302167bce4f64b60989a540b15bb620a"
          data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">
          Sign&nbsp;in
        </a>
          <a href="/join?source=header-repo&amp;source_repo=distillpub%2Fpost--momentum"
            class="HeaderMenu-link d-inline-block no-underline border border-gray-dark rounded-1 px-2 py-1"
            data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="0ea4411e45e08790b1f87371047ae103302167bce4f64b60989a540b15bb620a"
            data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">
            Sign&nbsp;up
          </a>
      </div>
    </div>
  </div>
</header>

  </div>

  <div id="start-of-content" class="show-on-focus"></div>


    <div id="js-flash-container">

</div>


      

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>




  <div class="application-main " data-commit-hovercards-enabled>
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main  >
      

  




  









  <div class="pagehead repohead hx_repohead readability-menu bg-gray-light pb-0 pt-0 pt-lg-3">

    <div class="d-flex container-lg mb-4 p-responsive d-none d-lg-flex">

      <div class="flex-auto min-width-0 width-fit mr-3">
        <h1 class="public  d-flex flex-wrap flex-items-center break-word float-none ">
    <svg class="octicon octicon-repo" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author ml-1 flex-self-stretch" itemprop="author">
    <a class="url fn" rel="author" data-hovercard-type="organization" data-hovercard-url="/orgs/distillpub/hovercard" href="/distillpub">distillpub</a>
  </span>
  <span class="path-divider flex-self-stretch">/</span>
  <strong itemprop="name" class="mr-2 flex-self-stretch">
    <a data-pjax="#js-repo-pjax-container" href="/distillpub/post--momentum">post--momentum</a>
  </strong>
  
</h1>


      </div>

      <ul class="pagehead-actions flex-shrink-0 " >




  <li>
    
  <a class="tooltipped tooltipped-s btn btn-sm btn-with-count" aria-label="You must be signed in to watch a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;notification subscription menu watch&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="0b8e07385e0e8548fe5bd774e5d431114619962fb512e863a5de672f719d1203" href="/login?return_to=%2Fdistillpub%2Fpost--momentum">
    <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
    Watch
</a>    <a class="social-count" href="/distillpub/post--momentum/watchers"
       aria-label="13 users are watching this repository">
      13
    </a>

  </li>

  <li>
        <a class="btn btn-sm btn-with-count tooltipped tooltipped-s" aria-label="You must be signed in to star a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;star button&quot;,&quot;repository_id&quot;:80573697,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="d025409b78f3bb095df162a2ccd6df3de00866c2cbbc872851a53bd65b65f506" href="/login?return_to=%2Fdistillpub%2Fpost--momentum">
      <svg height="16" class="octicon octicon-star v-align-text-bottom" vertical_align="text_bottom" viewBox="0 0 14 16" version="1.1" width="14" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>

      Star
</a>
    <a class="social-count js-social-count" href="/distillpub/post--momentum/stargazers"
      aria-label="160 users starred this repository">
      160
    </a>

  </li>

  <li>
      <a class="btn btn-sm btn-with-count tooltipped tooltipped-s" aria-label="You must be signed in to fork a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;repo details fork button&quot;,&quot;repository_id&quot;:80573697,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="94caeffaec0b7f08d68ed6b6312a6be3350afc27a18902a930bb984d9e0fc2ab" href="/login?return_to=%2Fdistillpub%2Fpost--momentum">
        <svg class="octicon octicon-repo-forked v-align-text-bottom" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 00-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 002 1a1.993 1.993 0 00-1 3.72V6.5l3 3v1.78A1.993 1.993 0 005 15a1.993 1.993 0 001-3.72V9.5l3-3V4.72A1.993 1.993 0 008 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
        Fork
</a>
    <a href="/distillpub/post--momentum/network/members" class="social-count"
       aria-label="48 users forked this repository">
      48
    </a>
  </li>
</ul>

    </div>
      
<nav class="hx_reponav reponav js-repo-nav js-sidenav-container-pjax clearfix container-lg p-responsive d-none d-lg-block"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
    aria-label="Repository"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a class="js-selected-navigation-item selected reponav-item" itemprop="url" data-hotkey="g c" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /distillpub/post--momentum" href="/distillpub/post--momentum">
      <div class="d-inline"><svg class="octicon octicon-code" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg></div>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" data-hotkey="g i" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /distillpub/post--momentum/issues" href="/distillpub/post--momentum/issues">
        <div class="d-inline"><svg class="octicon octicon-issue-opened" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 011.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg></div>
        <span itemprop="name">Issues</span>
        <span class="Counter">11</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a data-hotkey="g p" data-skip-pjax="true" itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /distillpub/post--momentum/pulls" href="/distillpub/post--momentum/pulls">
      <div class="d-inline"><svg class="octicon octicon-git-pull-request" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0010 15a1.993 1.993 0 001-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 00-1 3.72v6.56A1.993 1.993 0 002 15a1.993 1.993 0 001-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg></div>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">4</span>
      <meta itemprop="position" content="4">
</a>  </span>


    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement" class="position-relative float-left">
      <a data-hotkey="g w" data-skip-pjax="true" class="js-selected-navigation-item reponav-item" data-selected-links="repo_actions /distillpub/post--momentum/actions" href="/distillpub/post--momentum/actions">
        <div class="d-inline"><svg class="octicon octicon-play" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 8A7 7 0 110 8a7 7 0 0114 0zm-8.223 3.482l4.599-3.066a.5.5 0 000-.832L5.777 4.518A.5.5 0 005 4.934v6.132a.5.5 0 00.777.416z"/></svg></div>
        Actions
</a>
    </span>

    <a data-hotkey="g b" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /distillpub/post--momentum/projects" href="/distillpub/post--momentum/projects">
      <div class="d-inline"><svg class="octicon octicon-project" viewBox="0 0 15 16" version="1.1" width="15" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg></div>
      Projects
      <span class="Counter">0</span>
</a>

    <a data-skip-pjax="true" class="js-selected-navigation-item reponav-item" data-selected-links="security alerts policy token_scanning code_scanning /distillpub/post--momentum/security/advisories" href="/distillpub/post--momentum/security/advisories">
      <div class="d-inline"><svg class="octicon octicon-shield" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 2l7-2 7 2v6.02C14 12.69 8.69 16 7 16c-1.69 0-7-3.31-7-7.98V2zm1 .75L7 1l6 1.75v5.268C13 12.104 8.449 15 7 15c-1.449 0-6-2.896-6-6.982V2.75zm1 .75L7 2v12c-1.207 0-5-2.482-5-5.985V3.5z"/></svg></div>
      Security
</a>
    <a class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people /distillpub/post--momentum/pulse" href="/distillpub/post--momentum/pulse">
      <div class="d-inline"><svg class="octicon octicon-graph" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg></div>
      Insights
</a>

</nav>

  <div class="reponav-wrapper reponav-small d-lg-none">
  <nav class="reponav js-reponav text-center no-wrap"
       itemscope
       itemtype="http://schema.org/BreadcrumbList">

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a class="js-selected-navigation-item selected reponav-item" itemprop="url" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /distillpub/post--momentum" href="/distillpub/post--momentum">
        <span itemprop="name">Code</span>
        <meta itemprop="position" content="1">
</a>    </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /distillpub/post--momentum/issues" href="/distillpub/post--momentum/issues">
          <span itemprop="name">Issues</span>
          <span class="Counter">11</span>
          <meta itemprop="position" content="2">
</a>      </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /distillpub/post--momentum/pulls" href="/distillpub/post--momentum/pulls">
        <span itemprop="name">Pull requests</span>
        <span class="Counter">4</span>
        <meta itemprop="position" content="4">
</a>    </span>


      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /distillpub/post--momentum/projects" href="/distillpub/post--momentum/projects">
          <span itemprop="name">Projects</span>
          <span class="Counter">0</span>
          <meta itemprop="position" content="5">
</a>      </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_actions /distillpub/post--momentum/actions" href="/distillpub/post--momentum/actions">
          <span itemprop="name">Actions</span>
          <meta itemprop="position" content="6">
</a>      </span>


      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="security alerts policy token_scanning code_scanning /distillpub/post--momentum/security/advisories" href="/distillpub/post--momentum/security/advisories">
        <span itemprop="name">Security</span>
        <meta itemprop="position" content="8">
</a>
      <a class="js-selected-navigation-item reponav-item" data-selected-links="pulse /distillpub/post--momentum/pulse" href="/distillpub/post--momentum/pulse">
        Pulse
</a>

  </nav>
</div>


  </div>

  

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>


<div class="container-lg clearfix new-discussion-timeline  p-responsive">
  <div class="repository-content ">

    
    


  


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/distillpub/post--momentum/blob/691048b9d00b4b49b830c602b970755781df332c/public/assets/utils.js">Permalink</a>

    <!-- blob contrib key: blob_contributors:v22:d1031238a4606daf6b5126d61daaf5bd -->
      <div class="signup-prompt-bg rounded-1 js-signup-prompt" data-prompt="signup" hidden>
    <div class="signup-prompt p-4 text-center mb-4 rounded-1">
      <div class="position-relative">
        <button type="button" class="position-absolute top-0 right-0 btn-link link-gray js-signup-prompt-button" data-ga-click="(Logged out) Sign up prompt, clicked Dismiss, text:dismiss">
          Dismiss
        </button>
        <h3 class="pt-2">Join GitHub today</h3>
        <p class="col-6 mx-auto">GitHub is home to over 40 million developers working together to host and review code, manage projects, and build software together.</p>
        <a class="btn btn-primary" data-ga-click="(Logged out) Sign up prompt, clicked Sign up, text:sign-up" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;files signup prompt&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="f55ecc7fa912e23d5079beaf04fb305170ac0405c3e7b417e1bff5207aa4423a" href="/join?source=prompt-blob-show&amp;source_repo=distillpub%2Fpost--momentum">Sign up</a>
      </div>
    </div>
  </div>


    <div class="d-flex flex-items-start flex-shrink-0 flex-column flex-md-row pb-3">
      <span class="d-flex flex-justify-between width-full width-md-auto">
        
<details class="details-reset details-overlay branch-select-menu " id="branch-select-menu">
  <summary class="btn css-truncate btn-sm"
           data-hotkey="w"
           title="Switch branches or tags">
    <i>Branch:</i>
    <span class="css-truncate-target" data-menu-button>master</span>
    <span class="dropdown-caret"></span>
  </summary>

  <details-menu class="SelectMenu SelectMenu--hasFilter" src="/distillpub/post--momentum/refs/master/public/assets/utils.js?source_action=show&amp;source_controller=blob" preload>
    <div class="SelectMenu-modal">
      <include-fragment class="SelectMenu-loading" aria-label="Menu is loading">
        <svg class="octicon octicon-octoface anim-pulse" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M14.7 5.34c.13-.32.55-1.59-.13-3.31 0 0-1.05-.33-3.44 1.3-1-.28-2.07-.32-3.13-.32s-2.13.04-3.13.32c-2.39-1.64-3.44-1.3-3.44-1.3-.68 1.72-.26 2.99-.13 3.31C.49 6.21 0 7.33 0 8.69 0 13.84 3.33 15 7.98 15S16 13.84 16 8.69c0-1.36-.49-2.48-1.3-3.35zM8 14.02c-3.3 0-5.98-.15-5.98-3.35 0-.76.38-1.48 1.02-2.07 1.07-.98 2.9-.46 4.96-.46 2.07 0 3.88-.52 4.96.46.65.59 1.02 1.3 1.02 2.07 0 3.19-2.68 3.35-5.98 3.35zM5.49 9.01c-.66 0-1.2.8-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.54-1.78-1.2-1.78zm5.02 0c-.66 0-1.2.79-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.53-1.78-1.2-1.78z"/></svg>
      </include-fragment>
    </div>
  </details-menu>
</details>

        <div class="BtnGroup flex-shrink-0 d-md-none">
          <a href="/distillpub/post--momentum/find/master"
                class="js-pjax-capture-input btn btn-sm BtnGroup-item"
                data-pjax
                data-hotkey="t">
            Find file
          </a>
          <clipboard-copy value="public/assets/utils.js" class="btn btn-sm BtnGroup-item">
            Copy path
          </clipboard-copy>
        </div>
      </span>
      <h2 id="blob-path" class="breadcrumb flex-auto min-width-0 text-normal flex-md-self-center ml-md-2 mr-md-3 my-2 my-md-0">
          <span class="js-repo-root text-bold"><span class="js-path-segment"><a data-pjax="true" href="/distillpub/post--momentum"><span>post--momentum</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/distillpub/post--momentum/tree/master/public"><span>public</span></a></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/distillpub/post--momentum/tree/master/public/assets"><span>assets</span></a></span><span class="separator">/</span><strong class="final-path">utils.js</strong><span> /</span>
<details class="details-reset details-overlay d-inline" id="jumpto-symbol-select-menu">
  <summary class="btn-link link-gray css-truncate" aria-haspopup="true" data-hotkey="r" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.click_on_blob_definitions&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;click_on_blob_definitions&quot;,&quot;repository_id&quot;:80573697,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;JavaScript&quot;,&quot;originating_url&quot;:&quot;https://github.com/distillpub/post--momentum/blob/master/public/assets/utils.js&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="44005b16bff662bb11505804d7aed339f6baa317f4698088fbf68d5d13d67ac7">
      <svg class="octicon octicon-code" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
    <span data-menu-button>Jump to</span>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="SelectMenu SelectMenu--hasFilter" role="menu">
    <div class="SelectMenu-modal">
      <header class="SelectMenu-header">
        <span class="SelectMenu-title">Code definitions</span>
        <button class="SelectMenu-closeButton" type="button" data-toggle-for="jumpto-symbol-select-menu">
          <svg aria-label="Close menu" class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" role="img"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
        </button>
      </header>
      <div class="SelectMenu-list">
          <div class="SelectMenu-blankslate">
            <p class="mb-0 text-gray">
              No definitions found in this file.
            </p>
          </div>
        <div data-filterable-for="jumpto-symbols-filter-field" data-filterable-type="substring">
          
        </div>
      </div>
      <footer class="SelectMenu-footer">
        <div class="d-flex flex-justify-between">
          Unable to determine state of code navigation
          <svg class="octicon octicon-primitive-dot text-light-gray" viewBox="0 0 8 16" version="1.1" width="8" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 8c0-2.2 1.8-4 4-4s4 1.8 4 4-1.8 4-4 4-4-1.8-4-4z"/></svg>
        </div>
      </footer>
    </div>
  </details-menu>
</details>

      </h2>

      <div class="BtnGroup flex-shrink-0 d-none d-md-inline-block">
        <a href="/distillpub/post--momentum/find/master"
              class="js-pjax-capture-input btn btn-sm BtnGroup-item"
              data-pjax
              data-hotkey="t">
          Find file
        </a>
        <clipboard-copy value="public/assets/utils.js" class="btn btn-sm BtnGroup-item">
          Copy path
        </clipboard-copy>
      </div>
    </div>



    <include-fragment src="/distillpub/post--momentum/contributors/master/public/assets/utils.js" class="Box Box--condensed commit-loader">
      <div class="Box-body bg-blue-light f6">
        Fetching contributors&hellip;
      </div>

      <div class="Box-body d-flex flex-items-center" >
        <img alt="" class="loader-loading mr-2" src="https://github.githubassets.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" height="16" />
        <span class="text-red h6 loader-error">Cannot retrieve contributors at this time</span>
      </div>
</include-fragment>





    <div class="Box mt-3 position-relative
      ">
      
<div class="Box-header py-2 d-flex flex-column flex-shrink-0 flex-md-row flex-md-items-center">
  <div class="text-mono f6 flex-auto pr-3 flex-order-2 flex-md-order-1 mt-2 mt-md-0">

      1510 lines (1250 sloc)
      <span class="file-info-divider"></span>
    59.3 KB
  </div>

  <div class="d-flex py-1 py-md-0 flex-auto flex-order-1 flex-md-order-2 flex-sm-grow-0 flex-justify-between">

    <div class="BtnGroup">
      <a id="raw-url" class="btn btn-sm BtnGroup-item" href="/distillpub/post--momentum/raw/master/public/assets/utils.js">Raw</a>
        <a class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b" href="/distillpub/post--momentum/blame/master/public/assets/utils.js">Blame</a>
      <a rel="nofollow" class="btn btn-sm BtnGroup-item" href="/distillpub/post--momentum/commits/master/public/assets/utils.js">History</a>
    </div>


    <div>
          <a class="btn-octicon tooltipped tooltipped-nw js-remove-unless-platform"
             data-platforms="windows,mac"
             href="https://desktop.github.com"
             aria-label="Open this file in GitHub Desktop"
             data-ga-click="Repository, open with desktop">
              <svg class="octicon octicon-device-desktop" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M15 2H1c-.55 0-1 .45-1 1v9c0 .55.45 1 1 1h5.34c-.25.61-.86 1.39-2.34 2h8c-1.48-.61-2.09-1.39-2.34-2H15c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm0 9H1V3h14v8z"/></svg>
          </a>

          <button type="button" class="btn-octicon disabled tooltipped tooltipped-nw"
            aria-label="You must be signed in to make or propose changes">
            <svg class="octicon octicon-pencil" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 011.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
          </button>
          <button type="button" class="btn-octicon btn-octicon-danger disabled tooltipped tooltipped-nw"
            aria-label="You must be signed in to make or propose changes">
            <svg class="octicon octicon-trashcan" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
          </button>
    </div>
  </div>
</div>



      

  <div itemprop="text" class="Box-body p-0 blob-wrapper data type-javascript ">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8" data-paste-markdown-skip>
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/****************************************************************************</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  COLORMAPS AND STUFF</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class=pl-c>****************************************************************************/</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class=pl-k>var</span> <span class=pl-s1>colorbrewer</span><span class=pl-c1>=</span><span class=pl-kos>{</span><span class=pl-c1>YlGn</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcb9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#addd8e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2e699&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238443&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2e699&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#addd8e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#addd8e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238443&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005a32&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffe5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7fcb9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#addd8e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238443&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005a32&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffe5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7fcb9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#addd8e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#78c679&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238443&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006837&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#004529&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>YlGnBu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8b1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fcdbb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2c7fb8&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1dab4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#225ea8&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1dab4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2c7fb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#253494&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fcdbb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2c7fb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#253494&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fcdbb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1d91c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#225ea8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0c2c84&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#edf8b1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fcdbb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1d91c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#225ea8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0c2c84&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#edf8b1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fcdbb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41b6c4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1d91c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#225ea8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#253494&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#081d58&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>GnBu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e0f3db&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a8ddb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#43a2ca&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f0f9e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bae4bc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f0f9e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bae4bc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#43a2ca&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0868ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f0f9e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a8ddb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#43a2ca&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0868ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f0f9e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a8ddb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4eb3d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08589e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcf0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3db&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a8ddb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4eb3d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08589e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcf0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3db&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a8ddb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7bccc4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4eb3d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0868ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#084081&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>BuGn</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e5f5f9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d8c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2ca25f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2e2e2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2e2e2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2ca25f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccece6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d8c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2ca25f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccece6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d8c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ae76&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005824&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5f5f9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccece6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d8c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ae76&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005824&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5f5f9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccece6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d8c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ae76&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#00441b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PuBuGn</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ece2f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1c9099&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f6eff7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdc9e1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#02818a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f6eff7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdc9e1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1c9099&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#016c59&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f6eff7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1c9099&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#016c59&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f6eff7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#02818a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#016450&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ece2f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#02818a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#016450&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ece2f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#02818a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#016c59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#014636&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PuBu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ece7f2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdc9e1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0570b0&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdc9e1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#045a8d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b8cbe&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#045a8d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0570b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#034e7b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ece7f2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0570b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#034e7b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ece7f2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d0d1e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6bddb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3690c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0570b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#045a8d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#023858&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>BuPu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e0ecf4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ebcda&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8856a7&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#88419d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8856a7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#810f7c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bfd3e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ebcda&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8856a7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#810f7c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8fb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bfd3e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ebcda&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c6bb1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#88419d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6e016b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0ecf4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bfd3e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ebcda&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c6bb1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#88419d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6e016b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0ecf4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bfd3e6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ebcda&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c96c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c6bb1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#88419d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#810f7c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d004b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>RdPu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fde0dd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fa9fb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c51b8a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feebe2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fbb4b9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ae017e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feebe2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fbb4b9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c51b8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7a0177&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feebe2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcc5c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fa9fb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c51b8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7a0177&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feebe2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcc5c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fa9fb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd3497&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ae017e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7a0177&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7f3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0dd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcc5c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fa9fb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd3497&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ae017e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7a0177&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7f3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0dd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcc5c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fa9fb5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f768a1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd3497&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ae017e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7a0177&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#49006a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PuRd</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e7e1ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c994c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd1c77&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7b5d8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ce1256&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7b5d8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd1c77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#980043&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d4b9da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c994c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dd1c77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#980043&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1eef6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d4b9da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c994c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ce1256&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91003f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f4f9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7e1ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d4b9da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c994c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ce1256&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91003f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f4f9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7e1ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d4b9da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c994c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#df65b0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ce1256&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#980043&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67001f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>OrRd</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee8c8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbb84&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e34a33&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fef0d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcc8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7301f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fef0d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcc8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e34a33&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b30000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fef0d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd49e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbb84&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e34a33&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b30000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fef0d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd49e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbb84&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef6548&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7301f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#990000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7ec&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee8c8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd49e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbb84&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef6548&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7301f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#990000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7ec&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee8c8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd49e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbb84&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef6548&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d7301f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b30000&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7f0000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>YlOrRd</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffeda0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#feb24c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f03b20&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffb2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fecc5c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffb2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fecc5c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f03b20&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bd0026&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffb2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed976&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#feb24c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f03b20&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bd0026&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffb2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed976&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#feb24c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc4e2a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b10026&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffeda0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed976&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#feb24c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc4e2a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b10026&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffeda0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed976&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#feb24c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc4e2a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bd0026&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#800026&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>YlOrBr</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff7bc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fec44f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f0e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed98e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cc4c02&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed98e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f0e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#993404&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee391&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fec44f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f0e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#993404&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffd4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee391&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fec44f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ec7014&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cc4c02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c2d04&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffe5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fff7bc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee391&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fec44f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ec7014&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cc4c02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c2d04&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffe5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fff7bc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee391&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fec44f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fe9929&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ec7014&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cc4c02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#993404&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#662506&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Purples</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#efedf5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bcbddc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#756bb1&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f2f0f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbc9e2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a51a3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f2f0f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbc9e2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#756bb1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#54278f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f2f0f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dadaeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bcbddc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#756bb1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#54278f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f2f0f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dadaeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bcbddc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#807dba&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a51a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4a1486&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fcfbfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#efedf5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dadaeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bcbddc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#807dba&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a51a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4a1486&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fcfbfd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#efedf5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dadaeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bcbddc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9e9ac8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#807dba&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a51a3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#54278f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3f007d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Blues</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#deebf7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ecae1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3182bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#eff3ff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdd7e7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2171b5&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#eff3ff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdd7e7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3182bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08519c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#eff3ff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c6dbef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ecae1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3182bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08519c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#eff3ff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c6dbef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ecae1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4292c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2171b5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#084594&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fbff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#deebf7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c6dbef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ecae1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4292c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2171b5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#084594&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fbff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#deebf7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c6dbef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9ecae1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6baed6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4292c6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2171b5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08519c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#08306b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Greens</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e5f5e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d99b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bae4b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bae4b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d99b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#31a354&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#edf8e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d99b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005a32&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcf5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5f5e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d99b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#005a32&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7fcf5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5f5e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7e9c0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d99b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74c476&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#41ab5d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#238b45&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006d2c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#00441b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Oranges</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee6ce&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae6b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6550d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feedde&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbe85&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d94701&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feedde&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbe85&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6550d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a63603&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feedde&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd0a2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae6b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6550d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a63603&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#feedde&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd0a2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae6b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f16913&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d94801&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c2d04&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff5eb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee6ce&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd0a2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae6b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f16913&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d94801&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c2d04&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff5eb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee6ce&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdd0a2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae6b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fd8d3c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f16913&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d94801&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a63603&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7f2704&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Reds</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee0d2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc9272&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de2d26&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee5d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcae91&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cb181d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee5d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcae91&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de2d26&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a50f15&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee5d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcbba1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc9272&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de2d26&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a50f15&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fee5d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcbba1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc9272&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef3b2c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cb181d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99000d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0d2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcbba1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc9272&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef3b2c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cb181d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99000d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fff5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0d2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fcbba1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc9272&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb6a4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef3b2c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cb181d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a50f15&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67000d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Greys</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f0f0f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdbdbd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#636363&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cccccc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#525252&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cccccc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#636363&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#252525&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdbdbd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#636363&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#252525&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdbdbd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#737373&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#525252&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#252525&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f0f0f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdbdbd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#737373&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#525252&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#252525&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f0f0f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bdbdbd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#969696&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#737373&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#525252&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#252525&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#000000&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PuOr</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#f1a340&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#998ec3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e66101&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5e3c99&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e66101&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5e3c99&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1a340&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#998ec3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1a340&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#998ec3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e08214&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8073ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e08214&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8073ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7f3b08&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e08214&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8073ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2d004b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7f3b08&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b35806&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e08214&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb863&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee0b6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8daeb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2abd2&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8073ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#542788&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2d004b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>BrBG</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d8b365&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f5f5f5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5ab4ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6611a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#018571&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6611a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f5f5f5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#018571&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8b365&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5ab4ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d8b365&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f5f5f5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5ab4ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf812d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#35978f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf812d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f5f5f5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#35978f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#543005&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf812d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#35978f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#003c30&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#543005&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8c510a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf812d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#dfc27d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f6e8c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f5f5f5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c7eae5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80cdc1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#35978f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#01665e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#003c30&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PRGn</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#af8dc3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbf7b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7b3294&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#008837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7b3294&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#008837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#af8dc3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbf7b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#af8dc3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbf7b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9970ab&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5aae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9970ab&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5aae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#40004b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9970ab&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5aae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#00441b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#40004b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#762a83&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#9970ab&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c2a5cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7d4e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9f0d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6dba0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5aae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1b7837&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#00441b&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>PiYG</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e9a3c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d76a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d01c8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4dac26&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d01c8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4dac26&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e9a3c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d76a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e9a3c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a1d76a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de77ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbc41&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de77ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbc41&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8e0152&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de77ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbc41&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#276419&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8e0152&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#c51b7d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#de77ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1b6da&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fde0ef&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5d0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b8e186&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7fbc41&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d9221&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#276419&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>RdBu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ca0020&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0571b0&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ca0020&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#0571b0&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#67a9cf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4393c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4393c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#67001f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4393c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#053061&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#67001f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f7f7f7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d1e5f0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#92c5de&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4393c3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2166ac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#053061&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>RdGy</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#999999&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ca0020&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#404040&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#ca0020&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#404040&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#999999&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ef8a62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#999999&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#878787&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#878787&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#67001f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#878787&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a1a1a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#67001f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2182b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d6604d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4a582&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddbc7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffff&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0e0e0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bababa&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#878787&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4d4d4d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a1a1a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>RdYlBu</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91bfdb&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2c7bb6&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2c7bb6&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91bfdb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91bfdb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74add1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74add1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a50026&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74add1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#313695&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a50026&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee090&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e0f3f8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abd9e9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#74add1&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4575b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#313695&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Spectral</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d594&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b83ba&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b83ba&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d594&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#99d594&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#9e0142&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5e4fa2&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#9e0142&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d53e4f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f598&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#abdda4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#3288bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#5e4fa2&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>RdYlGn</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91cf60&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9641&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9641&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91cf60&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d59&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#91cf60&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66bd63&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66bd63&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a50026&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66bd63&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a50026&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d73027&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f46d43&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdae61&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fee08b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9ef8b&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d96a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66bd63&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1a9850&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#006837&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Accent</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#386cb0&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#386cb0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f0027f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#386cb0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f0027f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf5b17&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#7fc97f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#beaed4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdc086&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#386cb0&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f0027f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bf5b17&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#666666&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Dark2</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66a61e&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66a61e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6ab02&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66a61e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6ab02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6761d&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#1b9e77&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d95f02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#7570b3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e7298a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#66a61e&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6ab02&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6761d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#666666&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Paired</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cab2d6&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cab2d6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a3d9a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cab2d6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a3d9a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>12</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#a6cee3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#1f78b4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b2df8a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#33a02c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb9a99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e31a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdbf6f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cab2d6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#6a3d9a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff99&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b15928&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Pastel1</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed9a6&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed9a6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed9a6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5d8bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed9a6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5d8bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddaec&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#fbb4ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3cde3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#decbe4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fed9a6&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffcc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5d8bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fddaec&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f2f2f2&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Pastel2</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4cae4&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4cae4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5c9&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4cae4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fff2ae&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4cae4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fff2ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1e2cc&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#b3e2cd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdcdac&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cbd5e8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f4cae4&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e6f5c9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fff2ae&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f1e2cc&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#cccccc&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Set1</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff33&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff33&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a65628&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff33&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a65628&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f781bf&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#e41a1c&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#377eb8&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#4daf4a&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#984ea3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ff7f00&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffff33&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a65628&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#f781bf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#999999&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Set2</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e78ac3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e78ac3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d854&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e78ac3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d854&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffd92f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e78ac3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d854&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffd92f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5c494&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#66c2a5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fc8d62&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#8da0cb&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e78ac3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#a6d854&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffd92f&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#e5c494&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3b3b3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>,</span><span class=pl-c1>Set3</span>:<span class=pl-kos>{</span><span class=pl-c1>3</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>4</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>5</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>6</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>7</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>8</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fccde5&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>9</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fccde5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>10</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fccde5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bc80bd&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>11</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fccde5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bc80bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>12</span>:<span class=pl-kos>[</span><span class=pl-s>&quot;#8dd3c7&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffffb3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bebada&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fb8072&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#80b1d3&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fdb462&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#b3de69&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#fccde5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#d9d9d9&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#bc80bd&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ccebc5&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#ffed6f&quot;</span><span class=pl-kos>]</span><span class=pl-kos>}</span><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class=pl-k>var</span> <span class=pl-s1>jetc</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>100</span><span class=pl-kos>,</span><span class=pl-c1>1.5</span><span class=pl-kos>,</span><span class=pl-c1>2</span><span class=pl-kos>,</span><span class=pl-c1>3</span><span class=pl-kos>,</span><span class=pl-c1>4</span><span class=pl-kos>,</span><span class=pl-c1>5</span><span class=pl-kos>,</span><span class=pl-c1>10</span><span class=pl-kos>,</span><span class=pl-c1>60</span><span class=pl-kos>,</span><span class=pl-c1>200</span><span class=pl-kos>,</span><span class=pl-c1>500</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-s1>colorbrewer</span><span class=pl-kos>.</span><span class=pl-c1>RdYlBu</span><span class=pl-kos>[</span><span class=pl-c1>10</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class=pl-k>var</span> <span class=pl-s1>divergent</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>0.03</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-c1>0.03</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s>&quot;#d7191c&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;#ffffbf&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#2b83ba&quot;</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/****************************************************************************</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  &quot;GENERIC&quot; WIDGETS!</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line"><span class=pl-c>****************************************************************************/</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Custom slider with ticks, tooltips and all that jazz. */</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>sliderGen</span><span class=pl-kos>(</span><span class=pl-s1>dims</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-en>onMouseover</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-en>onMouseout</span>  <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-en>onChange</span>    <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ticks</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>margin</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-c1>right</span>: <span class=pl-c1>50</span><span class=pl-kos>,</span> <span class=pl-c1>left</span>: <span class=pl-c1>50</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>curr_xval</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>tooltipcallback</span> <span class=pl-c1>=</span> undefined</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>cr</span> <span class=pl-c1>=</span> <span class=pl-c1>9</span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>tickwidth</span> <span class=pl-c1>=</span> <span class=pl-c1>1.5</span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>tickheight</span> <span class=pl-c1>=</span> <span class=pl-c1>7</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ticksym</span> <span class=pl-c1>=</span> <span class=pl-c1>false</span> <span class=pl-c>// |---|-- vs |____|__</span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>shifty</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-c1>10</span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-en>ticktitles</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>round</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>showticks</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span></td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>default_xval</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>renderSlider</span><span class=pl-kos>(</span><span class=pl-s1>divin</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>tip</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>tip</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;class&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;d3-tip&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>offset</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>12</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>minLambda</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-c1>min</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-s1>ticks</span><span class=pl-kos>.</span><span class=pl-en>filter</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> !<span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>maxLambda</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-c1>max</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-s1>ticks</span><span class=pl-kos>.</span><span class=pl-en>filter</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> !<span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>width</span> <span class=pl-c1>=</span> <span class=pl-s1>dims</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span></td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>height</span> <span class=pl-c1>=</span> <span class=pl-s1>dims</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>svg</span> <span class=pl-c1>=</span> <span class=pl-s1>divin</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">	                  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>dims</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">	                  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span><span class=pl-s1>dims</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">	                  <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;relative&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">	                  <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">	                  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(0,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>shifty</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>maxLambda</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>clamp</span><span class=pl-kos>(</span><span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>slidersvg</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;slidersvg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>height</span> / <span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>slidersvg</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>tip</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>dragger</span> <span class=pl-c1>=</span> slidersvg.append(&quot;line&quot;)</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">        .<span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;track&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-en>appendChild</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>cloneNode</span><span class=pl-kos>(</span><span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;track-inset&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-en>appendChild</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>cloneNode</span><span class=pl-kos>(</span><span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;track-overlay&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;start.interrupt&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>slidersvg</span><span class=pl-kos>.</span><span class=pl-en>interrupt</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;start drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>var</span> <span class=pl-s1>xval</span> <span class=pl-c1>=</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>invert</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-c1>event</span><span class=pl-kos>.</span><span class=pl-c1>x</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,0)&quot;</span> <span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>curr_xval</span> <span class=pl-c1>=</span> <span class=pl-s1>xval</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>onChange</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>,</span> <span class=pl-s1>handle</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>ticksvg</span> <span class=pl-c1>=</span> <span class=pl-s1>slidersvg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>showticks</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">	    ticksvg.selectAll(&quot;rect&quot;)</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">	      .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>ticks</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;rect&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>9</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>tickwidth</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span> <span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-kos>(</span><span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>)</span> ? <span class=pl-c1>0</span>: <span class=pl-s1>ticksym</span> ? <span class=pl-s1>tickheight</span>*<span class=pl-c1>2</span>: <span class=pl-s1>tickheight</span><span class=pl-kos>;</span><span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0.2</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">	    ticksvg.selectAll(&quot;text&quot;)</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">	      .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>ticks</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">				  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;ticktext&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">				  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">				  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-anchor&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;middle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">			      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>tickwidth</span>*<span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-c1>24</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">			      <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-en>ticktitles</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">    ticksvg.selectAll(&quot;circle&quot;)</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">      .data(ticks,function(d,i) {return i})</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">      .enter()</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">      .<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;.track-overlay&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>;</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;mouseover&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>setAttribute</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;0.2&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-kos>(</span><span class=pl-s1>tooltipcallback</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>var</span> <span class=pl-s1>tooltip</span> <span class=pl-c1>=</span> <span class=pl-s1>tooltipcallback</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>tooltip</span> != <span class=pl-c1>false</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>tip</span><span class=pl-kos>.</span><span class=pl-en>show</span><span class=pl-kos>(</span><span class=pl-s>&#39;&lt;span&gt;&#39;</span> <span class=pl-c1>+</span> <span class=pl-s1>tooltip</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;&lt;/span&gt;&#39;</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>onMouseover</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;mouseout&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-kos>(</span><span class=pl-s1>tooltipcallback</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>tip</span><span class=pl-kos>.</span><span class=pl-en>hide</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">        <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>setAttribute</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;0&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>onMouseout</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;click&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>xval</span> <span class=pl-c1>=</span> <span class=pl-s1>lambda</span></td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>curr_xval</span> <span class=pl-c1>=</span> <span class=pl-s1>xval</span></td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,0)&quot;</span> <span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>onChange</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>,</span> <span class=pl-s1>handle</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line"><span class=pl-c>      Update the ticks</span></td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line"><span class=pl-c>    */</span></td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-en>updateTicks</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>newticks</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>d1</span> <span class=pl-c1>=</span> <span class=pl-s1>ticksvg</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;rect&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>newticks</span><span class=pl-kos>,</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d1</span><span class=pl-kos>.</span><span class=pl-en>exit</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>remove</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d1</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>d1</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>transition</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>duration</span><span class=pl-kos>(</span><span class=pl-c1>50</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-c1>0.5</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>d2</span> <span class=pl-c1>=</span> <span class=pl-s1>ticksvg</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>newticks</span><span class=pl-kos>,</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d2</span><span class=pl-kos>.</span><span class=pl-en>exit</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>remove</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d2</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>d2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>;</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>handle</span> <span class=pl-c1>=</span> <span class=pl-s1>slidersvg</span><span class=pl-kos>.</span><span class=pl-en>insert</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;.track-overlay&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>curr_xval</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,0)&quot;</span> <span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>insert</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;handle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>cr</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;#ff6600&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill-opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;white&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;start.interrupt&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>slidersvg</span><span class=pl-kos>.</span><span class=pl-en>interrupt</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;start drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>var</span> <span class=pl-s1>xval</span> <span class=pl-c1>=</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>invert</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-s1>dragger</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,0)&quot;</span> <span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>curr_xval</span> <span class=pl-c1>=</span> <span class=pl-s1>xval</span></td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>onChange</span><span class=pl-kos>(</span><span class=pl-s1>xval</span><span class=pl-kos>,</span> <span class=pl-s1>handle</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>insert</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(0,22)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-anchor&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;middle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;font-size&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;10px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>moveToFront</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-en>xval</span> : <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>curr_xval</span> <span class=pl-kos>}</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">    		<span class=pl-c1>tick</span> : <span class=pl-en>updateTicks</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">    		<span class=pl-en>init</span>:<span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">		        <span class=pl-s1>handle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>default_xval</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,0)&quot;</span> <span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">		        <span class=pl-en>onChange</span><span class=pl-kos>(</span><span class=pl-s1>default_xval</span><span class=pl-kos>,</span> <span class=pl-s1>handle</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">    		<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>ticktitles</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>ticktitles</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span></td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>mouseover</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>onMouseover</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span></td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>mouseout</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>onMouseOut</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span></td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>change</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>onChange</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span></td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>margin</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>margin</span> <span class=pl-c1>=</span> <span class=pl-s1>m</span></td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>ticks</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ticks</span> <span class=pl-c1>=</span> <span class=pl-s1>m</span></td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>startxval</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>curr_xval</span> <span class=pl-c1>=</span> <span class=pl-s1>m</span></td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>default_xval</span> <span class=pl-c1>=</span> <span class=pl-s1>m</span></td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>margins</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>l</span><span class=pl-kos>,</span> <span class=pl-s1>r</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>margin</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-c1>right</span>: <span class=pl-s1>l</span><span class=pl-kos>,</span> <span class=pl-c1>left</span>: <span class=pl-s1>r</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>tooltip</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tooltipcallback</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span></td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>cRadius</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cr</span> <span class=pl-c1>=</span> <span class=pl-s1>m</span></td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>tickConfig</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_1</span><span class=pl-kos>,</span><span class=pl-s1>_2</span><span class=pl-kos>,</span><span class=pl-s1>_3</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tickwidth</span> <span class=pl-c1>=</span> <span class=pl-s1>_1</span></td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tickheight</span> <span class=pl-c1>=</span> <span class=pl-s1>_2</span></td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ticksym</span> <span class=pl-c1>=</span> <span class=pl-s1>_3</span> <span class=pl-c>// |---|-- vs |____|__</span></td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>shifty</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>shifty</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>showticks</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>showticks</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderSlider</span><span class=pl-kos>.</span><span class=pl-en>shifty</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>shifty</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>renderSlider</span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Generate &quot;stick&quot; graph */</span></td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>stemGraphGen</span><span class=pl-kos>(</span><span class=pl-s1>graphWidth</span><span class=pl-kos>,</span> <span class=pl-s1>graphHeight</span><span class=pl-kos>,</span> <span class=pl-s1>n</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>borderTop</span> <span class=pl-c1>=</span> <span class=pl-c1>20</span></td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>borderLeft</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-c1>5</span></td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>axis</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ylabel</span> <span class=pl-c1>=</span> <span class=pl-v>MathCache</span><span class=pl-kos>(</span><span class=pl-s>&quot;x-i-k&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ylabelsize</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;13px&quot;</span></td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>r1</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span></td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>r2</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ticks</span> <span class=pl-c1>=</span> <span class=pl-c1>10</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>renderGraph</span><span class=pl-kos>(</span><span class=pl-s1>outdiv</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">    outdiv.append(&quot;span&quot;)</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">      .style(&quot;top&quot;, (graphHeight/2 + borderTop/2) + &quot;px&quot;)</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">      .style(&quot;left&quot;, (-graphHeight/2 - 17) + &quot;px&quot; )</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">      .<span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;absolute&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>graphHeight</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;20px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;absolute&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;rotate(-90deg)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-align&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;center&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;font-size&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>ylabelsize</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-s1>ylabel</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>svg</span> <span class=pl-c1>=</span> <span class=pl-s1>outdiv</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>graphWidth</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>graphHeight</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">          <span class=pl-c>// .style(&quot;border&quot;, &quot;black solid 1px&quot;)</span></td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">          <span class=pl-c>// .style(&quot;box-shadow&quot;,&quot;0px 0px 10px rgba(0, 0, 0, 0.2)&quot;)</span></td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;absolute&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;top&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>borderTop</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;left&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>borderLeft</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;border-radius&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;2px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>n</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>10</span><span class=pl-kos>,</span> <span class=pl-s1>graphWidth</span><span class=pl-c1>-</span><span class=pl-c1>10</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-s1>axis</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>graphHeight</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>cscale</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>axis</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>axis</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;black&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>valueline</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>line</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>y</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span>   <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// Initialize the data</span></td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-en>initData</span><span class=pl-kos>(</span><span class=pl-s1>color</span><span class=pl-kos>,</span> <span class=pl-s1>r</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>dots</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>dotsdata</span> <span class=pl-c1>=</span> <span class=pl-s1>dots</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-en>zeros</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span> <span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>i</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>dotsdata</span><span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>r</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;darkblue&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line">      dotsdata.enter()</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line">        .<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s1>color</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span><span class=pl-c1>1.5</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-s1>dots</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>dots1</span> <span class=pl-c1>=</span> <span class=pl-en>initData</span><span class=pl-kos>(</span><span class=pl-s>&quot;#999&quot;</span><span class=pl-kos>,</span><span class=pl-s1>r1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>dots2</span> <span class=pl-c1>=</span> <span class=pl-en>initData</span><span class=pl-kos>(</span><span class=pl-s1>colorbrewer</span><span class=pl-kos>.</span><span class=pl-c1>RdPu</span><span class=pl-kos>[</span><span class=pl-c1>5</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>2</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>r2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>function</span> <span class=pl-en>updateData</span><span class=pl-kos>(</span><span class=pl-s1>dots</span><span class=pl-kos>,</span> <span class=pl-s1>data</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>dots</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>data</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-c1>20</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>dots</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>data</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-c1>20</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-en>updatePath</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d1</span><span class=pl-kos>,</span> <span class=pl-s1>d2</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line">      <span class=pl-en>updateData</span><span class=pl-kos>(</span><span class=pl-s1>dots1</span><span class=pl-kos>,</span> <span class=pl-s1>d1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-kos>(</span><span class=pl-s1>d2</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-en>updateData</span><span class=pl-kos>(</span><span class=pl-s1>dots2</span><span class=pl-kos>,</span> <span class=pl-s1>d2</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// Add x axis</span></td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;grid&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(0,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>axisBottom</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>ticks</span><span class=pl-kos>(</span><span class=pl-s1>ticks</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>tickSize</span><span class=pl-kos>(</span><span class=pl-c1>2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>tickFormat</span><span class=pl-kos>(</span><span class=pl-s>&quot;&quot;</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-en>updatePath</span></td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>borderTop</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>borderTop</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderGraph</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>axis</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axis</span> <span class=pl-c1>=</span> <span class=pl-s1>a</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>ylabel</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ylabel</span> <span class=pl-c1>=</span> <span class=pl-s1>a</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>radius1</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>r1</span> <span class=pl-c1>=</span> <span class=pl-s1>a</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>labelSize</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>s</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>ylabelsize</span> <span class=pl-c1>=</span> <span class=pl-s1>s</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderGraph</span><span class=pl-kos>.</span><span class=pl-en>numTicks</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>s</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>ticks</span> <span class=pl-c1>=</span> <span class=pl-s1>s</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>renderGraph</span></td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Render a stacked graph. D*/</span></td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>stackedBarchartGen</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>,</span> <span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>axis</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-c1>1.53</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>translatex</span> <span class=pl-c1>=</span> <span class=pl-c1>110</span></td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>translatey</span> <span class=pl-c1>=</span> <span class=pl-c1>10</span></td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>col</span> <span class=pl-c1>=</span> <span class=pl-s1>colorbrewer</span><span class=pl-kos>.</span><span class=pl-c1>RdPu</span></td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>highlightcol</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;darkred&quot;</span></td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>lineopacity</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>cr</span> <span class=pl-c1>=</span> <span class=pl-c1>1.75</span></td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>copacity</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>dotcolor</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;black&quot;</span></td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>drawgrid</span> <span class=pl-c1>=</span> <span class=pl-c1>true</span></td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>renderStackedGraph</span><span class=pl-kos>(</span><span class=pl-s1>svg</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>dwidth</span>  <span class=pl-c1>=</span> <span class=pl-c1>800</span></td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>dheight</span> <span class=pl-c1>=</span> <span class=pl-c1>170</span></td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>margin</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span><span class=pl-c1>right</span>: <span class=pl-c1>23</span><span class=pl-kos>,</span> <span class=pl-c1>left</span>: <span class=pl-c1>10</span><span class=pl-kos>,</span> <span class=pl-c1>top</span>: <span class=pl-c1>10</span><span class=pl-kos>,</span> <span class=pl-c1>bottom</span>: <span class=pl-c1>10</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>width</span>  <span class=pl-c1>=</span> <span class=pl-s1>dwidth</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span></td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>height</span> <span class=pl-c1>=</span> <span class=pl-s1>dheight</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>top</span> <span class=pl-c1>-</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>bottom</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>graphsvg</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>translatex</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>translatey</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>stack</span> <span class=pl-c1>=</span> <span class=pl-en>zeros2D</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>,</span><span class=pl-s1>m</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>axisheight</span> <span class=pl-c1>=</span> <span class=pl-c1>10</span></td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>stack</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span><span class=pl-kos>,</span><span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span> <span class=pl-c1>+</span> <span class=pl-s1>width</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>axis</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>axis</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>height</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>function</span> <span class=pl-en>add</span><span class=pl-kos>(</span><span class=pl-s1>a</span><span class=pl-kos>,</span> <span class=pl-s1>b</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>a</span> <span class=pl-c1>+</span> <span class=pl-s1>b</span><span class=pl-kos>;</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>var</span> <span class=pl-s1>s</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>m</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line">		  <span class=pl-k>var</span> <span class=pl-s1>si</span> <span class=pl-c1>=</span> <span class=pl-s1>graphsvg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line">		  si.selectAll(&quot;line&quot;)</td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line">		    .data(stack)</td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line">		    .enter()</td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line">		    .<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span><span class=pl-c1>2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">		    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>col</span><span class=pl-kos>[</span><span class=pl-c1>3</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>lineopacity</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line">		  <span class=pl-s1>s</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>si</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">		graphsvg.append(&quot;g&quot;).<span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>stack</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-en>reduce</span><span class=pl-kos>(</span><span class=pl-s1>add</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>copacity</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>function</span> <span class=pl-en>updateGraph</span><span class=pl-kos>(</span><span class=pl-s1>stacknew</span><span class=pl-kos>,</span> <span class=pl-s1>highlight</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">			<span class=pl-k>var</span> <span class=pl-s1>svgdata</span> <span class=pl-c1>=</span> <span class=pl-s1>graphsvg</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>stacknew</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">			<span class=pl-s1>svgdata</span><span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line">			<span class=pl-s1>svgdata</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>svgdata</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line">			  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line">			  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-en>reduce</span><span class=pl-kos>(</span><span class=pl-s1>add</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line">			  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span>  <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>highlight</span><span class=pl-kos>.</span><span class=pl-en>includes</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>2</span> : <span class=pl-s1>cr</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line">			  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>highlight</span><span class=pl-kos>.</span><span class=pl-en>includes</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-s1>highlightcol</span> : <span class=pl-s1>dotcolor</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line">			<span class=pl-s1>svgdata</span><span class=pl-kos>.</span><span class=pl-en>exit</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>remove</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">			<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>m</span><span class=pl-kos>;</span> <span class=pl-s1>j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">		    <span class=pl-k>var</span> <span class=pl-s1>svgdatai</span> <span class=pl-c1>=</span> <span class=pl-s1>s</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>stacknew</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">		    <span class=pl-s1>svgdatai</span><span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">		    <span class=pl-s1>svgdatai</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>svgdatai</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">			    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-en>slice</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>j</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>reduce</span><span class=pl-kos>(</span><span class=pl-s1>add</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">			    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-en>slice</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>j</span><span class=pl-c1>+</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>reduce</span><span class=pl-kos>(</span><span class=pl-s1>add</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">		    <span class=pl-s1>svgdatai</span><span class=pl-kos>.</span><span class=pl-en>exit</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>remove</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line">			<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">		<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>drawgrid</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">  		<span class=pl-s1>graphsvg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;grid&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(0,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>height</span><span class=pl-c1>+</span><span class=pl-c1>10</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.25</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>axisBottom</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">  			  <span class=pl-kos>.</span><span class=pl-en>ticks</span><span class=pl-kos>(</span><span class=pl-c1>5</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">  			  <span class=pl-kos>.</span><span class=pl-en>tickSize</span><span class=pl-kos>(</span><span class=pl-c1>2</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">  		<span class=pl-s1>graphsvg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;grid&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(12,0)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.25</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">  			<span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>axisLeft</span><span class=pl-kos>(</span><span class=pl-v>Y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">  			  <span class=pl-kos>.</span><span class=pl-en>ticks</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">  			  <span class=pl-kos>.</span><span class=pl-en>tickSize</span><span class=pl-kos>(</span><span class=pl-c1>2</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">		<span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-c1>update</span>: <span class=pl-s1>updateGraph</span><span class=pl-kos>,</span> <span class=pl-c1>stack</span>: <span class=pl-s1>s</span><span class=pl-kos>,</span> <span class=pl-c1>X</span>:<span class=pl-v>X</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>highlightcol</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>highlightcol</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>translatex</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>translatex</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>translatey</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>translatey</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>col</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>col</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">  	<span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>lineopacity</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>lineopacity</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>cr</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cr</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>copacity</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>copacity</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>dotcolor</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>dotcolor</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>renderStackedGraph</span><span class=pl-kos>.</span><span class=pl-en>drawgrid</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>drawgrid</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span></td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>renderStackedGraph</span></td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* 2d &quot;scatterplot&quot; with lines generator */</span></td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>plot2dGen</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>,</span> <span class=pl-v>Y</span><span class=pl-kos>,</span> <span class=pl-s1>iterColor</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>cradius</span> <span class=pl-c1>=</span> <span class=pl-c1>1.2</span></td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>copacity</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>pathopacity</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>pathwidth</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>strokecolor</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;black&quot;</span></td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>plot2d</span><span class=pl-kos>(</span><span class=pl-s1>svg</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>svgpath</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;path&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>pathopacity</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;none&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>strokecolor</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span><span class=pl-s1>pathwidth</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-linecap&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;round&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>valueline</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>line</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>y</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>svgcircle</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-en>update</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-v>W</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">        <span class=pl-c>// Update Circles</span></td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>svgdata</span> <span class=pl-c1>=</span> <span class=pl-s1>svgcircle</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-v>W</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">        svgdata.<span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>cradius</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;box-shadow&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;0px 3px 10px rgba(0, 0, 0, 0.4)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>copacity</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>iterColor</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>svgdata</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>svgdata</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>cradius</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>copacity</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>iterColor</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>svgdata</span><span class=pl-kos>.</span><span class=pl-en>exit</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>remove</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">        <span class=pl-c>// Update Path</span></td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>svgpath</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;d&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>valueline</span><span class=pl-kos>(</span><span class=pl-v>W</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>return</span> <span class=pl-en>update</span></td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// var cradius = 1.2</span></td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// var copacity = 1</span></td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// var pathopacity = 1</span></td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// var pathwidth = 1</span></td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>circleRadius</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cradius</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>stroke</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>strokecolor</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>circleOpacity</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>copacity</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>pathWidth</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>pathwidth</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>pathOpacity</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>pathopacity</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>plot2d</span><span class=pl-kos>.</span><span class=pl-en>stroke</span> <span class=pl-c1>=</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>strokecolor</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>plot2d</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>plot2d</span></td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Render heatmap of f with colormap cmap onto canvas*/</span></td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>renderHeatmap</span><span class=pl-kos>(</span><span class=pl-s1>canvas</span><span class=pl-kos>,</span> <span class=pl-s1>f</span><span class=pl-kos>,</span> <span class=pl-s1>cmap</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>canvasWidth</span>  <span class=pl-c1>=</span> <span class=pl-s1>canvas</span><span class=pl-kos>.</span><span class=pl-c1>width</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>canvasHeight</span> <span class=pl-c1>=</span> <span class=pl-s1>canvas</span><span class=pl-kos>.</span><span class=pl-c1>height</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ctx</span> <span class=pl-c1>=</span> <span class=pl-s1>canvas</span><span class=pl-kos>.</span><span class=pl-en>getContext</span><span class=pl-kos>(</span><span class=pl-s>&#39;2d&#39;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>imageData</span> <span class=pl-c1>=</span> <span class=pl-s1>ctx</span><span class=pl-kos>.</span><span class=pl-en>getImageData</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>canvasWidth</span><span class=pl-kos>,</span> <span class=pl-s1>canvasHeight</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>buf</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>ArrayBuffer</span><span class=pl-kos>(</span><span class=pl-s1>imageData</span><span class=pl-kos>.</span><span class=pl-c1>data</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>buf8</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>Uint8ClampedArray</span><span class=pl-kos>(</span><span class=pl-s1>buf</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-v>Uint32Array</span><span class=pl-kos>(</span><span class=pl-s1>buf</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>canvasHeight</span><span class=pl-kos>;</span> <span class=pl-c1>++</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>canvasWidth</span><span class=pl-kos>;</span> <span class=pl-c1>++</span><span class=pl-s1>j</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">          <span class=pl-k>var</span> <span class=pl-s1>value</span> <span class=pl-c1>=</span> <span class=pl-en>parseColor</span><span class=pl-kos>(</span><span class=pl-s1>cmap</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>(</span><span class=pl-s1>j</span>/<span class=pl-s1>canvasWidth</span><span class=pl-kos>,</span> <span class=pl-s1>i</span>/<span class=pl-s1>canvasHeight</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">          <span class=pl-s1>data</span><span class=pl-kos>[</span><span class=pl-s1>i</span> * <span class=pl-s1>canvasWidth</span> <span class=pl-c1>+</span> <span class=pl-s1>j</span><span class=pl-kos>]</span> <span class=pl-c1>=</span></td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>(</span><span class=pl-c1>255</span> <span class=pl-c1>&lt;&lt;</span> <span class=pl-c1>24</span><span class=pl-kos>)</span> |        <span class=pl-c>// alpha</span></td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>(</span>~~<span class=pl-s1>value</span><span class=pl-kos>[</span><span class=pl-c1>2</span><span class=pl-kos>]</span> <span class=pl-c1>&lt;&lt;</span> <span class=pl-c1>16</span><span class=pl-kos>)</span> | <span class=pl-c>// blue</span></td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>(</span>~~<span class=pl-s1>value</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>&lt;&lt;</span>  <span class=pl-c1>8</span><span class=pl-kos>)</span> | <span class=pl-c>// green</span></td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line">              ~~<span class=pl-s1>value</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>;</span>          <span class=pl-c>// red</span></td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>imageData</span><span class=pl-kos>.</span><span class=pl-c1>data</span><span class=pl-kos>.</span><span class=pl-en>set</span><span class=pl-kos>(</span><span class=pl-s1>buf8</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>ctx</span><span class=pl-kos>.</span><span class=pl-en>putImageData</span><span class=pl-kos>(</span><span class=pl-s1>imageData</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>slider2D</span><span class=pl-kos>(</span><span class=pl-s1>div</span><span class=pl-kos>,</span> <span class=pl-s1>onChange</span><span class=pl-kos>,</span> <span class=pl-s1>lambda1</span><span class=pl-kos>,</span> <span class=pl-s1>lambdan</span><span class=pl-kos>,</span> <span class=pl-s1>start</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>panel</span> <span class=pl-c1>=</span> <span class=pl-s1>div</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">	              <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">	              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(25,30)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>width</span> <span class=pl-c1>=</span> <span class=pl-c1>105</span></td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>maxX</span> <span class=pl-c1>=</span> <span class=pl-c1>4</span></td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>maxY</span> <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>maxX</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>2</span>*<span class=pl-s1>width</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>clamp</span><span class=pl-kos>(</span><span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>maxY</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>clamp</span><span class=pl-kos>(</span><span class=pl-c1>true</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>path</span> <span class=pl-c1>=</span> panel.<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;path&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;d&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;M 0 0 L &quot;</span> <span class=pl-c1>+</span> <span class=pl-c1>2</span>*<span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s>&quot; 0 L &quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s>&quot; &quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s>&quot; L 0 &quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s>&quot; z&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;#EEE&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;#EEE&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>5</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-linejoin&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;round&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;click&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">	              <span class=pl-k>var</span> <span class=pl-s1>pt</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">	              <span class=pl-k>var</span> <span class=pl-s1>xy</span> <span class=pl-c1>=</span> <span class=pl-en>clip</span><span class=pl-kos>(</span><span class=pl-s1>pt</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L678" class="blob-num js-line-number" data-line-number="678"></td>
        <td id="LC678" class="blob-code blob-code-inner js-file-line">	              <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L679" class="blob-num js-line-number" data-line-number="679"></td>
        <td id="LC679" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L680" class="blob-num js-line-number" data-line-number="680"></td>
        <td id="LC680" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L681" class="blob-num js-line-number" data-line-number="681"></td>
        <td id="LC681" class="blob-code blob-code-inner js-file-line">	              <span class=pl-k>var</span> <span class=pl-s1>pt</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L682" class="blob-num js-line-number" data-line-number="682"></td>
        <td id="LC682" class="blob-code blob-code-inner js-file-line">	              <span class=pl-k>var</span> <span class=pl-s1>xy</span> <span class=pl-c1>=</span> <span class=pl-en>clip</span><span class=pl-kos>(</span><span class=pl-s1>pt</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L683" class="blob-num js-line-number" data-line-number="683"></td>
        <td id="LC683" class="blob-code blob-code-inner js-file-line">	              <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L684" class="blob-num js-line-number" data-line-number="684"></td>
        <td id="LC684" class="blob-code blob-code-inner js-file-line">	            <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L685" class="blob-num js-line-number" data-line-number="685"></td>
        <td id="LC685" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L686" class="blob-num js-line-number" data-line-number="686"></td>
        <td id="LC686" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>ly</span> <span class=pl-c1>=</span> <span class=pl-s1>panel</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L687" class="blob-num js-line-number" data-line-number="687"></td>
        <td id="LC687" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L688" class="blob-num js-line-number" data-line-number="688"></td>
        <td id="LC688" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L689" class="blob-num js-line-number" data-line-number="689"></td>
        <td id="LC689" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span><span class=pl-s1>width</span>*<span class=pl-c1>2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L690" class="blob-num js-line-number" data-line-number="690"></td>
        <td id="LC690" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L691" class="blob-num js-line-number" data-line-number="691"></td>
        <td id="LC691" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;#DDD&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L692" class="blob-num js-line-number" data-line-number="692"></td>
        <td id="LC692" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;3px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L693" class="blob-num js-line-number" data-line-number="693"></td>
        <td id="LC693" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L694" class="blob-num js-line-number" data-line-number="694"></td>
        <td id="LC694" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>lx</span> <span class=pl-c1>=</span> <span class=pl-s1>panel</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L695" class="blob-num js-line-number" data-line-number="695"></td>
        <td id="LC695" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L696" class="blob-num js-line-number" data-line-number="696"></td>
        <td id="LC696" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L697" class="blob-num js-line-number" data-line-number="697"></td>
        <td id="LC697" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L698" class="blob-num js-line-number" data-line-number="698"></td>
        <td id="LC698" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span><span class=pl-s1>width</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L699" class="blob-num js-line-number" data-line-number="699"></td>
        <td id="LC699" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;#DDD&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L700" class="blob-num js-line-number" data-line-number="700"></td>
        <td id="LC700" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;3px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L701" class="blob-num js-line-number" data-line-number="701"></td>
        <td id="LC701" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L702" class="blob-num js-line-number" data-line-number="702"></td>
        <td id="LC702" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>xval</span> <span class=pl-c1>=</span> <span class=pl-c1>10</span></td>
      </tr>
      <tr>
        <td id="L703" class="blob-num js-line-number" data-line-number="703"></td>
        <td id="LC703" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>yval</span> <span class=pl-c1>=</span> <span class=pl-c1>10</span></td>
      </tr>
      <tr>
        <td id="L704" class="blob-num js-line-number" data-line-number="704"></td>
        <td id="LC704" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L705" class="blob-num js-line-number" data-line-number="705"></td>
        <td id="LC705" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>function</span> <span class=pl-en>clip</span><span class=pl-kos>(</span><span class=pl-s1>pt</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L706" class="blob-num js-line-number" data-line-number="706"></td>
        <td id="LC706" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>max</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>pt</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-s1>width</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L707" class="blob-num js-line-number" data-line-number="707"></td>
        <td id="LC707" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>max</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>pt</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-c1>2</span>*<span class=pl-s1>width</span> <span class=pl-c1>-</span> <span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L708" class="blob-num js-line-number" data-line-number="708"></td>
        <td id="LC708" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L709" class="blob-num js-line-number" data-line-number="709"></td>
        <td id="LC709" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L710" class="blob-num js-line-number" data-line-number="710"></td>
        <td id="LC710" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L711" class="blob-num js-line-number" data-line-number="711"></td>
        <td id="LC711" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>function</span> <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L712" class="blob-num js-line-number" data-line-number="712"></td>
        <td id="LC712" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>circle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span><span class=pl-s1>x</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L713" class="blob-num js-line-number" data-line-number="713"></td>
        <td id="LC713" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>circle</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L714" class="blob-num js-line-number" data-line-number="714"></td>
        <td id="LC714" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>ly</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span><span class=pl-c1>2</span>*<span class=pl-s1>width</span><span class=pl-c1>-</span><span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L715" class="blob-num js-line-number" data-line-number="715"></td>
        <td id="LC715" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>lx</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-c1>&gt;</span><span class=pl-s1>width</span><span class=pl-kos>)</span> ? <span class=pl-kos>(</span><span class=pl-s1>width</span> <span class=pl-c1>-</span> <span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-c1>-</span><span class=pl-s1>width</span><span class=pl-kos>)</span><span class=pl-kos>)</span> : <span class=pl-s1>width</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L716" class="blob-num js-line-number" data-line-number="716"></td>
        <td id="LC716" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>onChange</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>.</span><span class=pl-en>invert</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-v>Y</span><span class=pl-kos>.</span><span class=pl-en>invert</span><span class=pl-kos>(</span><span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L717" class="blob-num js-line-number" data-line-number="717"></td>
        <td id="LC717" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L718" class="blob-num js-line-number" data-line-number="718"></td>
        <td id="LC718" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L719" class="blob-num js-line-number" data-line-number="719"></td>
        <td id="LC719" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>circle</span> <span class=pl-c1>=</span> <span class=pl-s1>panel</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>7</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;rgb(255, 102, 0)&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;white&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span></td>
      </tr>
      <tr>
        <td id="L720" class="blob-num js-line-number" data-line-number="720"></td>
        <td id="LC720" class="blob-code blob-code-inner js-file-line">	    <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L721" class="blob-num js-line-number" data-line-number="721"></td>
        <td id="LC721" class="blob-code blob-code-inner js-file-line">	      <span class=pl-k>var</span> <span class=pl-s1>pt</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L722" class="blob-num js-line-number" data-line-number="722"></td>
        <td id="LC722" class="blob-code blob-code-inner js-file-line">	      <span class=pl-k>var</span> <span class=pl-s1>xy</span> <span class=pl-c1>=</span> <span class=pl-en>clip</span><span class=pl-kos>(</span><span class=pl-s1>pt</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L723" class="blob-num js-line-number" data-line-number="723"></td>
        <td id="LC723" class="blob-code blob-code-inner js-file-line">	      <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L724" class="blob-num js-line-number" data-line-number="724"></td>
        <td id="LC724" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L725" class="blob-num js-line-number" data-line-number="725"></td>
        <td id="LC725" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L726" class="blob-num js-line-number" data-line-number="726"></td>
        <td id="LC726" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>tickwidth</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span></td>
      </tr>
      <tr>
        <td id="L727" class="blob-num js-line-number" data-line-number="727"></td>
        <td id="LC727" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>tickheight</span> <span class=pl-c1>=</span> <span class=pl-c1>6</span></td>
      </tr>
      <tr>
        <td id="L728" class="blob-num js-line-number" data-line-number="728"></td>
        <td id="LC728" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>tickgroupX</span> <span class=pl-c1>=</span> <span class=pl-s1>panel</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(0, &quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-c1>12</span><span class=pl-kos>)</span> <span class=pl-c1>+</span><span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L729" class="blob-num js-line-number" data-line-number="729"></td>
        <td id="LC729" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L730" class="blob-num js-line-number" data-line-number="730"></td>
        <td id="LC730" class="blob-code blob-code-inner js-file-line">	tickgroupX.selectAll(&quot;rect&quot;)</td>
      </tr>
      <tr>
        <td id="L731" class="blob-num js-line-number" data-line-number="731"></td>
        <td id="LC731" class="blob-code blob-code-inner js-file-line">	  .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>maxX</span>/<span class=pl-c1>2</span><span class=pl-kos>,</span><span class=pl-s1>maxX</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L732" class="blob-num js-line-number" data-line-number="732"></td>
        <td id="LC732" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;rect&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L733" class="blob-num js-line-number" data-line-number="733"></td>
        <td id="LC733" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L734" class="blob-num js-line-number" data-line-number="734"></td>
        <td id="LC734" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L735" class="blob-num js-line-number" data-line-number="735"></td>
        <td id="LC735" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>tickwidth</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L736" class="blob-num js-line-number" data-line-number="736"></td>
        <td id="LC736" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>tickheight</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L737" class="blob-num js-line-number" data-line-number="737"></td>
        <td id="LC737" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0.2</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L738" class="blob-num js-line-number" data-line-number="738"></td>
        <td id="LC738" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L739" class="blob-num js-line-number" data-line-number="739"></td>
        <td id="LC739" class="blob-code blob-code-inner js-file-line">	tickgroupX.selectAll(&quot;text&quot;)</td>
      </tr>
      <tr>
        <td id="L740" class="blob-num js-line-number" data-line-number="740"></td>
        <td id="LC740" class="blob-code blob-code-inner js-file-line">	  .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>maxX</span>/<span class=pl-c1>2</span><span class=pl-kos>,</span><span class=pl-s1>maxX</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L741" class="blob-num js-line-number" data-line-number="741"></td>
        <td id="LC741" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L742" class="blob-num js-line-number" data-line-number="742"></td>
        <td id="LC742" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;ticktext&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L743" class="blob-num js-line-number" data-line-number="743"></td>
        <td id="LC743" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L744" class="blob-num js-line-number" data-line-number="744"></td>
        <td id="LC744" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-anchor&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;middle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L745" class="blob-num js-line-number" data-line-number="745"></td>
        <td id="LC745" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>tickwidth</span>*<span class=pl-c1>2</span> <span class=pl-c1>-</span><span class=pl-c1>8</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L746" class="blob-num js-line-number" data-line-number="746"></td>
        <td id="LC746" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>d</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L747" class="blob-num js-line-number" data-line-number="747"></td>
        <td id="LC747" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L748" class="blob-num js-line-number" data-line-number="748"></td>
        <td id="LC748" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>tickgroupY</span> <span class=pl-c1>=</span> <span class=pl-s1>panel</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-c1>-</span><span class=pl-c1>12</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;, 0)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L749" class="blob-num js-line-number" data-line-number="749"></td>
        <td id="LC749" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L750" class="blob-num js-line-number" data-line-number="750"></td>
        <td id="LC750" class="blob-code blob-code-inner js-file-line">	tickgroupY.selectAll(&quot;rect&quot;)</td>
      </tr>
      <tr>
        <td id="L751" class="blob-num js-line-number" data-line-number="751"></td>
        <td id="LC751" class="blob-code blob-code-inner js-file-line">	  .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>maxY</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L752" class="blob-num js-line-number" data-line-number="752"></td>
        <td id="LC752" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;rect&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L753" class="blob-num js-line-number" data-line-number="753"></td>
        <td id="LC753" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L754" class="blob-num js-line-number" data-line-number="754"></td>
        <td id="LC754" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-en>isNaN</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> ? <span class=pl-c1>-</span><span class=pl-c1>100</span>: <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L755" class="blob-num js-line-number" data-line-number="755"></td>
        <td id="LC755" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>tickheight</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L756" class="blob-num js-line-number" data-line-number="756"></td>
        <td id="LC756" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>tickwidth</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L757" class="blob-num js-line-number" data-line-number="757"></td>
        <td id="LC757" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0.2</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L758" class="blob-num js-line-number" data-line-number="758"></td>
        <td id="LC758" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L759" class="blob-num js-line-number" data-line-number="759"></td>
        <td id="LC759" class="blob-code blob-code-inner js-file-line">	tickgroupY.selectAll(&quot;text&quot;)</td>
      </tr>
      <tr>
        <td id="L760" class="blob-num js-line-number" data-line-number="760"></td>
        <td id="LC760" class="blob-code blob-code-inner js-file-line">	  .<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>maxY</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-k>function</span>(<span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>i</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L761" class="blob-num js-line-number" data-line-number="761"></td>
        <td id="LC761" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L762" class="blob-num js-line-number" data-line-number="762"></td>
        <td id="LC762" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;ticktext&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L763" class="blob-num js-line-number" data-line-number="763"></td>
        <td id="LC763" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L764" class="blob-num js-line-number" data-line-number="764"></td>
        <td id="LC764" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-anchor&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;middle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L765" class="blob-num js-line-number" data-line-number="765"></td>
        <td id="LC765" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-c1>8</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-s1>tickwidth</span>/<span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-c1>5</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L766" class="blob-num js-line-number" data-line-number="766"></td>
        <td id="LC766" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>d</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L767" class="blob-num js-line-number" data-line-number="767"></td>
        <td id="LC767" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L768" class="blob-num js-line-number" data-line-number="768"></td>
        <td id="LC768" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>beta</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambda1</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambdan</span><span class=pl-kos>)</span><span class=pl-kos>)</span>/<span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambda1</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambdan</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>beta</span> <span class=pl-c1>=</span> <span class=pl-s1>beta</span>*<span class=pl-s1>beta</span></td>
      </tr>
      <tr>
        <td id="L769" class="blob-num js-line-number" data-line-number="769"></td>
        <td id="LC769" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>alpha</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span>/<span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambda1</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>lambdan</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L770" class="blob-num js-line-number" data-line-number="770"></td>
        <td id="LC770" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>alpha</span> <span class=pl-c1>=</span> <span class=pl-s1>alpha</span>*<span class=pl-s1>alpha</span></td>
      </tr>
      <tr>
        <td id="L771" class="blob-num js-line-number" data-line-number="771"></td>
        <td id="LC771" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L772" class="blob-num js-line-number" data-line-number="772"></td>
        <td id="LC772" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>specialpoints</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>[</span><span class=pl-s1>alpha</span>*<span class=pl-s1>lambdan</span><span class=pl-kos>,</span> <span class=pl-s1>beta</span><span class=pl-kos>]</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L773" class="blob-num js-line-number" data-line-number="773"></td>
        <td id="LC773" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L774" class="blob-num js-line-number" data-line-number="774"></td>
        <td id="LC774" class="blob-code blob-code-inner js-file-line">	panel.append(&quot;g&quot;).selectAll(&quot;circle&quot;)</td>
      </tr>
      <tr>
        <td id="L775" class="blob-num js-line-number" data-line-number="775"></td>
        <td id="LC775" class="blob-code blob-code-inner js-file-line">	  .data(specialpoints)</td>
      </tr>
      <tr>
        <td id="L776" class="blob-num js-line-number" data-line-number="776"></td>
        <td id="LC776" class="blob-code blob-code-inner js-file-line">	  .<span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;circle&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L777" class="blob-num js-line-number" data-line-number="777"></td>
        <td id="LC777" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L778" class="blob-num js-line-number" data-line-number="778"></td>
        <td id="LC778" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L779" class="blob-num js-line-number" data-line-number="779"></td>
        <td id="LC779" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>2</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L780" class="blob-num js-line-number" data-line-number="780"></td>
        <td id="LC780" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0.2</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L781" class="blob-num js-line-number" data-line-number="781"></td>
        <td id="LC781" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;cursor&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;pointer&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L782" class="blob-num js-line-number" data-line-number="782"></td>
        <td id="LC782" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;click&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L783" class="blob-num js-line-number" data-line-number="783"></td>
        <td id="LC783" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L784" class="blob-num js-line-number" data-line-number="784"></td>
        <td id="LC784" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>if</span>  <span class=pl-kos>(</span>!<span class=pl-kos>(</span><span class=pl-s1>start</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L785" class="blob-num js-line-number" data-line-number="785"></td>
        <td id="LC785" class="blob-code blob-code-inner js-file-line">		<span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>start</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>start</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L786" class="blob-num js-line-number" data-line-number="786"></td>
        <td id="LC786" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L787" class="blob-num js-line-number" data-line-number="787"></td>
        <td id="LC787" class="blob-code blob-code-inner js-file-line">	  <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>alpha</span>*<span class=pl-s1>lambdan</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>beta</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L788" class="blob-num js-line-number" data-line-number="788"></td>
        <td id="LC788" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L789" class="blob-num js-line-number" data-line-number="789"></td>
        <td id="LC789" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L790" class="blob-num js-line-number" data-line-number="790"></td>
        <td id="LC790" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>circle</span><span class=pl-kos>.</span><span class=pl-en>moveToFront</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L791" class="blob-num js-line-number" data-line-number="791"></td>
        <td id="LC791" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L792" class="blob-num js-line-number" data-line-number="792"></td>
        <td id="LC792" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-en>changeMouse</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>(</span><span class=pl-s1>specialpoints</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-v>Y</span><span class=pl-kos>(</span><span class=pl-s1>specialpoints</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L793" class="blob-num js-line-number" data-line-number="793"></td>
        <td id="LC793" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L794" class="blob-num js-line-number" data-line-number="794"></td>
        <td id="LC794" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L795" class="blob-num js-line-number" data-line-number="795"></td>
        <td id="LC795" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/****************************************************************************</span></td>
      </tr>
      <tr>
        <td id="L796" class="blob-num js-line-number" data-line-number="796"></td>
        <td id="LC796" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  OPTIMIZATION RELATED FUNCTIONS</span></td>
      </tr>
      <tr>
        <td id="L797" class="blob-num js-line-number" data-line-number="797"></td>
        <td id="LC797" class="blob-code blob-code-inner js-file-line"><span class=pl-c>****************************************************************************/</span></td>
      </tr>
      <tr>
        <td id="L798" class="blob-num js-line-number" data-line-number="798"></td>
        <td id="LC798" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L799" class="blob-num js-line-number" data-line-number="799"></td>
        <td id="LC799" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L800" class="blob-num js-line-number" data-line-number="800"></td>
        <td id="LC800" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Solves the system (1 - Lambda[i]*alpha)^k &lt;= 1e-7</span></td>
      </tr>
      <tr>
        <td id="L801" class="blob-num js-line-number" data-line-number="801"></td>
        <td id="LC801" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  for k.</span></td>
      </tr>
      <tr>
        <td id="L802" class="blob-num js-line-number" data-line-number="802"></td>
        <td id="LC802" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L803" class="blob-num js-line-number" data-line-number="803"></td>
        <td id="LC803" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>getStepsConvergence</span><span class=pl-kos>(</span><span class=pl-v>Lambda</span><span class=pl-kos>,</span> <span class=pl-s1>alpha</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L804" class="blob-num js-line-number" data-line-number="804"></td>
        <td id="LC804" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-v>Lambda</span><span class=pl-kos>.</span><span class=pl-en>map</span><span class=pl-kos>(</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>lambdai</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L805" class="blob-num js-line-number" data-line-number="805"></td>
        <td id="LC805" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>o</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-c1>3</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span>/<span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>log10</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span> <span class=pl-s1>lambdai</span>*<span class=pl-s1>alpha</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L806" class="blob-num js-line-number" data-line-number="806"></td>
        <td id="LC806" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>o</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>0</span> ? <span class=pl-v>NaN</span> : <span class=pl-s1>o</span></td>
      </tr>
      <tr>
        <td id="L807" class="blob-num js-line-number" data-line-number="807"></td>
        <td id="LC807" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L808" class="blob-num js-line-number" data-line-number="808"></td>
        <td id="LC808" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L809" class="blob-num js-line-number" data-line-number="809"></td>
        <td id="LC809" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L810" class="blob-num js-line-number" data-line-number="810"></td>
        <td id="LC810" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L811" class="blob-num js-line-number" data-line-number="811"></td>
        <td id="LC811" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Run Momentum the good old fashioned way - by iterating.</span></td>
      </tr>
      <tr>
        <td id="L812" class="blob-num js-line-number" data-line-number="812"></td>
        <td id="LC812" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  &gt; runMomentum(bananaf, [0,0], 0.00001, 0.5, 100)</span></td>
      </tr>
      <tr>
        <td id="L813" class="blob-num js-line-number" data-line-number="813"></td>
        <td id="LC813" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L814" class="blob-num js-line-number" data-line-number="814"></td>
        <td id="LC814" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>runMomentum</span><span class=pl-kos>(</span><span class=pl-s1>f</span><span class=pl-kos>,</span> <span class=pl-s1>w0</span><span class=pl-kos>,</span> <span class=pl-s1>alpha</span><span class=pl-kos>,</span> <span class=pl-s1>beta</span><span class=pl-kos>,</span> <span class=pl-s1>totalIters</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L815" class="blob-num js-line-number" data-line-number="815"></td>
        <td id="LC815" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Obj</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-v>W</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-s1>z</span> <span class=pl-c1>=</span> <span class=pl-en>zeros</span><span class=pl-kos>(</span><span class=pl-s1>w0</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-s1>w</span> <span class=pl-c1>=</span> <span class=pl-s1>w0</span></td>
      </tr>
      <tr>
        <td id="L816" class="blob-num js-line-number" data-line-number="816"></td>
        <td id="LC816" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fx</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span><span class=pl-kos>(</span><span class=pl-s1>w0</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-s1>gx</span> <span class=pl-c1>=</span> <span class=pl-s1>fx</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L817" class="blob-num js-line-number" data-line-number="817"></td>
        <td id="LC817" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>W</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>w0</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-v>Obj</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>fx</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L818" class="blob-num js-line-number" data-line-number="818"></td>
        <td id="LC818" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>totalIters</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L819" class="blob-num js-line-number" data-line-number="819"></td>
        <td id="LC819" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>z</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>add</span><span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-s1>beta</span><span class=pl-kos>,</span> <span class=pl-s1>z</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-s1>gx</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L820" class="blob-num js-line-number" data-line-number="820"></td>
        <td id="LC820" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>w</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>add</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>,</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-s1>alpha</span><span class=pl-kos>,</span> <span class=pl-s1>z</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L821" class="blob-num js-line-number" data-line-number="821"></td>
        <td id="LC821" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fx</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>gx</span> <span class=pl-c1>=</span> <span class=pl-s1>fx</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L822" class="blob-num js-line-number" data-line-number="822"></td>
        <td id="LC822" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>.</span><span class=pl-en>every</span><span class=pl-kos>(</span><span class=pl-s1>isFinite</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L823" class="blob-num js-line-number" data-line-number="823"></td>
        <td id="LC823" class="blob-code blob-code-inner js-file-line">      <span class=pl-v>W</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-v>Obj</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>fx</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L824" class="blob-num js-line-number" data-line-number="824"></td>
        <td id="LC824" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span><span class=pl-kos>{</span> <span class=pl-k>break</span><span class=pl-kos>;</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L825" class="blob-num js-line-number" data-line-number="825"></td>
        <td id="LC825" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L826" class="blob-num js-line-number" data-line-number="826"></td>
        <td id="LC826" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-v>Obj</span><span class=pl-kos>,</span> <span class=pl-v>W</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L827" class="blob-num js-line-number" data-line-number="827"></td>
        <td id="LC827" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L828" class="blob-num js-line-number" data-line-number="828"></td>
        <td id="LC828" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L829" class="blob-num js-line-number" data-line-number="829"></td>
        <td id="LC829" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L830" class="blob-num js-line-number" data-line-number="830"></td>
        <td id="LC830" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Closed form solution to the gradient descent iteration</span></td>
      </tr>
      <tr>
        <td id="L831" class="blob-num js-line-number" data-line-number="831"></td>
        <td id="LC831" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L832" class="blob-num js-line-number" data-line-number="832"></td>
        <td id="LC832" class="blob-code blob-code-inner js-file-line"><span class=pl-c>w+ = w - alpha*([U*Lambda*U&#39;]*w - b)</span></td>
      </tr>
      <tr>
        <td id="L833" class="blob-num js-line-number" data-line-number="833"></td>
        <td id="LC833" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L834" class="blob-num js-line-number" data-line-number="834"></td>
        <td id="LC834" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Usage</span></td>
      </tr>
      <tr>
        <td id="L835" class="blob-num js-line-number" data-line-number="835"></td>
        <td id="LC835" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L836" class="blob-num js-line-number" data-line-number="836"></td>
        <td id="LC836" class="blob-code blob-code-inner js-file-line"><span class=pl-c>iter = getiter(U,Lambda, b, alpha)</span></td>
      </tr>
      <tr>
        <td id="L837" class="blob-num js-line-number" data-line-number="837"></td>
        <td id="LC837" class="blob-code blob-code-inner js-file-line"><span class=pl-c>w    = iter(1000) &lt;- gets the 1000&#39;th iteration.</span></td>
      </tr>
      <tr>
        <td id="L838" class="blob-num js-line-number" data-line-number="838"></td>
        <td id="LC838" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L839" class="blob-num js-line-number" data-line-number="839"></td>
        <td id="LC839" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>geniter</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>,</span> <span class=pl-v>Lambda</span><span class=pl-kos>,</span> <span class=pl-s1>b</span><span class=pl-kos>,</span> <span class=pl-s1>alpha</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L840" class="blob-num js-line-number" data-line-number="840"></td>
        <td id="LC840" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L841" class="blob-num js-line-number" data-line-number="841"></td>
        <td id="LC841" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Ub</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>,</span><span class=pl-s1>b</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L842" class="blob-num js-line-number" data-line-number="842"></td>
        <td id="LC842" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L843" class="blob-num js-line-number" data-line-number="843"></td>
        <td id="LC843" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>c</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L844" class="blob-num js-line-number" data-line-number="844"></td>
        <td id="LC844" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-v>U</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L845" class="blob-num js-line-number" data-line-number="845"></td>
        <td id="LC845" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>var</span> <span class=pl-s1>ci</span> <span class=pl-c1>=</span> <span class=pl-v>Ub</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-c1>1</span> <span class=pl-c1>-</span> <span class=pl-s1>alpha</span>*<span class=pl-v>Lambda</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span><span class=pl-kos>)</span>/<span class=pl-v>Lambda</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L846" class="blob-num js-line-number" data-line-number="846"></td>
        <td id="LC846" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>c</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>ci</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L847" class="blob-num js-line-number" data-line-number="847"></td>
        <td id="LC847" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L848" class="blob-num js-line-number" data-line-number="848"></td>
        <td id="LC848" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>c</span><span class=pl-kos>,</span> <span class=pl-v>U</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L849" class="blob-num js-line-number" data-line-number="849"></td>
        <td id="LC849" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L850" class="blob-num js-line-number" data-line-number="850"></td>
        <td id="LC850" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L851" class="blob-num js-line-number" data-line-number="851"></td>
        <td id="LC851" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L852" class="blob-num js-line-number" data-line-number="852"></td>
        <td id="LC852" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L853" class="blob-num js-line-number" data-line-number="853"></td>
        <td id="LC853" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Generates function which computes the matrix geometric series</span></td>
      </tr>
      <tr>
        <td id="L854" class="blob-num js-line-number" data-line-number="854"></td>
        <td id="LC854" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  sum([A^i for i = 0:(k-1)])*b</span></td>
      </tr>
      <tr>
        <td id="L855" class="blob-num js-line-number" data-line-number="855"></td>
        <td id="LC855" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L856" class="blob-num js-line-number" data-line-number="856"></td>
        <td id="LC856" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>matSum</span><span class=pl-kos>(</span><span class=pl-v>R</span><span class=pl-kos>,</span><span class=pl-s1>b</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L857" class="blob-num js-line-number" data-line-number="857"></td>
        <td id="LC857" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L858" class="blob-num js-line-number" data-line-number="858"></td>
        <td id="LC858" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// Buxfix hack for buggy numeric.js code.</span></td>
      </tr>
      <tr>
        <td id="L859" class="blob-num js-line-number" data-line-number="859"></td>
        <td id="LC859" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>fix</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L860" class="blob-num js-line-number" data-line-number="860"></td>
        <td id="LC860" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>]</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L861" class="blob-num js-line-number" data-line-number="861"></td>
        <td id="LC861" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-k>typeof</span> <span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> <span class=pl-s>&quot;number&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L862" class="blob-num js-line-number" data-line-number="862"></td>
        <td id="LC862" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-en>zeros</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L863" class="blob-num js-line-number" data-line-number="863"></td>
        <td id="LC863" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L864" class="blob-num js-line-number" data-line-number="864"></td>
        <td id="LC864" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-en>zeros2D</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>,</span> <span class=pl-v>U</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L865" class="blob-num js-line-number" data-line-number="865"></td>
        <td id="LC865" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L866" class="blob-num js-line-number" data-line-number="866"></td>
        <td id="LC866" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L867" class="blob-num js-line-number" data-line-number="867"></td>
        <td id="LC867" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L868" class="blob-num js-line-number" data-line-number="868"></td>
        <td id="LC868" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L869" class="blob-num js-line-number" data-line-number="869"></td>
        <td id="LC869" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// Complex Diag functionality</span></td>
      </tr>
      <tr>
        <td id="L870" class="blob-num js-line-number" data-line-number="870"></td>
        <td id="LC870" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-v>A</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L871" class="blob-num js-line-number" data-line-number="871"></td>
        <td id="LC871" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-v>A</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L872" class="blob-num js-line-number" data-line-number="872"></td>
        <td id="LC872" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-v>A</span><span class=pl-kos>[</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L873" class="blob-num js-line-number" data-line-number="873"></td>
        <td id="LC873" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-k>new</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-c1>T</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>,</span><span class=pl-v>Y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L874" class="blob-num js-line-number" data-line-number="874"></td>
        <td id="LC874" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L875" class="blob-num js-line-number" data-line-number="875"></td>
        <td id="LC875" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L876" class="blob-num js-line-number" data-line-number="876"></td>
        <td id="LC876" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// Hack for tacking powers of complex numbers</span></td>
      </tr>
      <tr>
        <td id="L877" class="blob-num js-line-number" data-line-number="877"></td>
        <td id="LC877" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L878" class="blob-num js-line-number" data-line-number="878"></td>
        <td id="LC878" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>log</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>exp</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L879" class="blob-num js-line-number" data-line-number="879"></td>
        <td id="LC879" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L880" class="blob-num js-line-number" data-line-number="880"></td>
        <td id="LC880" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L881" class="blob-num js-line-number" data-line-number="881"></td>
        <td id="LC881" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>eR</span>     <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>eig</span><span class=pl-kos>(</span><span class=pl-v>R</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L882" class="blob-num js-line-number" data-line-number="882"></td>
        <td id="LC882" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// U*lambda*inv(U) = R</span></td>
      </tr>
      <tr>
        <td id="L883" class="blob-num js-line-number" data-line-number="883"></td>
        <td id="LC883" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>lambda</span> <span class=pl-c1>=</span> <span class=pl-s1>eR</span><span class=pl-kos>[</span><span class=pl-s>&quot;lambda&quot;</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-en>fix</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L884" class="blob-num js-line-number" data-line-number="884"></td>
        <td id="LC884" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>U</span>      <span class=pl-c1>=</span> <span class=pl-s1>eR</span><span class=pl-kos>[</span><span class=pl-s>&quot;E&quot;</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-en>fix</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L885" class="blob-num js-line-number" data-line-number="885"></td>
        <td id="LC885" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>bc</span>     <span class=pl-c1>=</span> <span class=pl-k>new</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-c1>T</span><span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>b</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-en>zeros2D</span><span class=pl-kos>(</span><span class=pl-s1>b</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>,</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L886" class="blob-num js-line-number" data-line-number="886"></td>
        <td id="LC886" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Uinvb</span>  <span class=pl-c1>=</span> <span class=pl-v>U</span><span class=pl-kos>.</span><span class=pl-en>inv</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>bc</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-en>fix</span><span class=pl-kos>(</span><span class=pl-v>Uinvb</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L887" class="blob-num js-line-number" data-line-number="887"></td>
        <td id="LC887" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// console.log(numeric.prettyPrint(R))</span></td>
      </tr>
      <tr>
        <td id="L888" class="blob-num js-line-number" data-line-number="888"></td>
        <td id="LC888" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// console.log(numeric.prettyPrint(lambda))</span></td>
      </tr>
      <tr>
        <td id="L889" class="blob-num js-line-number" data-line-number="889"></td>
        <td id="LC889" class="blob-code blob-code-inner js-file-line">  <span class=pl-c>// console.log(1-numeric.norm2([lambda.y[0],lambda.x[0]]), 1-numeric.norm2([lambda.y[1],lambda.x[1]]))</span></td>
      </tr>
      <tr>
        <td id="L890" class="blob-num js-line-number" data-line-number="890"></td>
        <td id="LC890" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-en>matSum</span>: <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L891" class="blob-num js-line-number" data-line-number="891"></td>
        <td id="LC891" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>topv</span> <span class=pl-c1>=</span> <span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>,</span><span class=pl-s1>k</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>add</span><span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L892" class="blob-num js-line-number" data-line-number="892"></td>
        <td id="LC892" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>botv</span> <span class=pl-c1>=</span> <span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>add</span><span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L893" class="blob-num js-line-number" data-line-number="893"></td>
        <td id="LC893" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>sumk</span> <span class=pl-c1>=</span> <span class=pl-s1>topv</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-s1>botv</span><span class=pl-kos>,</span><span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L894" class="blob-num js-line-number" data-line-number="894"></td>
        <td id="LC894" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-en>getRow</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>sumk</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>k</span><span class=pl-kos>;</span> <span class=pl-s1>sumk</span><span class=pl-kos>[</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L895" class="blob-num js-line-number" data-line-number="895"></td>
        <td id="LC895" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-en>getRow</span><span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span> <span class=pl-c1>==</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>sumk</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>k</span><span class=pl-kos>;</span> <span class=pl-s1>sumk</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L896" class="blob-num js-line-number" data-line-number="896"></td>
        <td id="LC896" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-s1>sumk</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>Uinvb</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L897" class="blob-num js-line-number" data-line-number="897"></td>
        <td id="LC897" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>lambda</span>:<span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>norm2</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-c1>y</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-c1>x</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L898" class="blob-num js-line-number" data-line-number="898"></td>
        <td id="LC898" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L899" class="blob-num js-line-number" data-line-number="899"></td>
        <td id="LC899" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L900" class="blob-num js-line-number" data-line-number="900"></td>
        <td id="LC900" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L901" class="blob-num js-line-number" data-line-number="901"></td>
        <td id="LC901" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L902" class="blob-num js-line-number" data-line-number="902"></td>
        <td id="LC902" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Matrix power</span></td>
      </tr>
      <tr>
        <td id="L903" class="blob-num js-line-number" data-line-number="903"></td>
        <td id="LC903" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L904" class="blob-num js-line-number" data-line-number="904"></td>
        <td id="LC904" class="blob-code blob-code-inner js-file-line"><span class=pl-c>// function matPow(A) {</span></td>
      </tr>
      <tr>
        <td id="L905" class="blob-num js-line-number" data-line-number="905"></td>
        <td id="LC905" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L906" class="blob-num js-line-number" data-line-number="906"></td>
        <td id="LC906" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   function fix(U) {</span></td>
      </tr>
      <tr>
        <td id="L907" class="blob-num js-line-number" data-line-number="907"></td>
        <td id="LC907" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     if (U[&quot;y&quot;] === undefined) {</span></td>
      </tr>
      <tr>
        <td id="L908" class="blob-num js-line-number" data-line-number="908"></td>
        <td id="LC908" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//       if (typeof U[&quot;x&quot;][0] == &quot;number&quot;) {</span></td>
      </tr>
      <tr>
        <td id="L909" class="blob-num js-line-number" data-line-number="909"></td>
        <td id="LC909" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//         U[&quot;y&quot;] = zeros(U[&quot;x&quot;].length)</span></td>
      </tr>
      <tr>
        <td id="L910" class="blob-num js-line-number" data-line-number="910"></td>
        <td id="LC910" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//       } else {</span></td>
      </tr>
      <tr>
        <td id="L911" class="blob-num js-line-number" data-line-number="911"></td>
        <td id="LC911" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//         U[&quot;y&quot;] = zeros2D(U[&quot;x&quot;].length, U[&quot;x&quot;][0].length)</span></td>
      </tr>
      <tr>
        <td id="L912" class="blob-num js-line-number" data-line-number="912"></td>
        <td id="LC912" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//       }</span></td>
      </tr>
      <tr>
        <td id="L913" class="blob-num js-line-number" data-line-number="913"></td>
        <td id="LC913" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     }</span></td>
      </tr>
      <tr>
        <td id="L914" class="blob-num js-line-number" data-line-number="914"></td>
        <td id="LC914" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   }</span></td>
      </tr>
      <tr>
        <td id="L915" class="blob-num js-line-number" data-line-number="915"></td>
        <td id="LC915" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L916" class="blob-num js-line-number" data-line-number="916"></td>
        <td id="LC916" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   // Complex Diag functionality</span></td>
      </tr>
      <tr>
        <td id="L917" class="blob-num js-line-number" data-line-number="917"></td>
        <td id="LC917" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   function diag(A) {</span></td>
      </tr>
      <tr>
        <td id="L918" class="blob-num js-line-number" data-line-number="918"></td>
        <td id="LC918" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     var X = numeric.diag(A[&quot;x&quot;])</span></td>
      </tr>
      <tr>
        <td id="L919" class="blob-num js-line-number" data-line-number="919"></td>
        <td id="LC919" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     var Y = numeric.diag(A[&quot;y&quot;])</span></td>
      </tr>
      <tr>
        <td id="L920" class="blob-num js-line-number" data-line-number="920"></td>
        <td id="LC920" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     return new numeric.T(X,Y)</span></td>
      </tr>
      <tr>
        <td id="L921" class="blob-num js-line-number" data-line-number="921"></td>
        <td id="LC921" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   }</span></td>
      </tr>
      <tr>
        <td id="L922" class="blob-num js-line-number" data-line-number="922"></td>
        <td id="LC922" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L923" class="blob-num js-line-number" data-line-number="923"></td>
        <td id="LC923" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   // Hack for tacking powers of complex numbers</span></td>
      </tr>
      <tr>
        <td id="L924" class="blob-num js-line-number" data-line-number="924"></td>
        <td id="LC924" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   function pow(x,k) {</span></td>
      </tr>
      <tr>
        <td id="L925" class="blob-num js-line-number" data-line-number="925"></td>
        <td id="LC925" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//     return x.log().mul(k).exp()</span></td>
      </tr>
      <tr>
        <td id="L926" class="blob-num js-line-number" data-line-number="926"></td>
        <td id="LC926" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   }</span></td>
      </tr>
      <tr>
        <td id="L927" class="blob-num js-line-number" data-line-number="927"></td>
        <td id="LC927" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L928" class="blob-num js-line-number" data-line-number="928"></td>
        <td id="LC928" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   var eR     = numeric.eig(A)</span></td>
      </tr>
      <tr>
        <td id="L929" class="blob-num js-line-number" data-line-number="929"></td>
        <td id="LC929" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   // var lambda = eR[&quot;lambda&quot;]; fix(lambda)</span></td>
      </tr>
      <tr>
        <td id="L930" class="blob-num js-line-number" data-line-number="930"></td>
        <td id="LC930" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   // var U      = eR[&quot;E&quot;]; fix(U)</span></td>
      </tr>
      <tr>
        <td id="L931" class="blob-num js-line-number" data-line-number="931"></td>
        <td id="LC931" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   // var Uinv   = U.inv(); fix(Uinv)</span></td>
      </tr>
      <tr>
        <td id="L932" class="blob-num js-line-number" data-line-number="932"></td>
        <td id="LC932" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L933" class="blob-num js-line-number" data-line-number="933"></td>
        <td id="LC933" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   return {matPow: function(k) {</span></td>
      </tr>
      <tr>
        <td id="L934" class="blob-num js-line-number" data-line-number="934"></td>
        <td id="LC934" class="blob-code blob-code-inner js-file-line"><span class=pl-c>// //    var lambdak = pow(lambda,k)</span></td>
      </tr>
      <tr>
        <td id="L935" class="blob-num js-line-number" data-line-number="935"></td>
        <td id="LC935" class="blob-code blob-code-inner js-file-line"><span class=pl-c>// //    return U.dot(diag(lambdak)).dot(Uinv)[&quot;x&quot;]</span></td>
      </tr>
      <tr>
        <td id="L936" class="blob-num js-line-number" data-line-number="936"></td>
        <td id="LC936" class="blob-code blob-code-inner js-file-line"><span class=pl-c>//   }, lambda:(numeric.norm2([lambda.y[0],lambda.x[0]]))}</span></td>
      </tr>
      <tr>
        <td id="L937" class="blob-num js-line-number" data-line-number="937"></td>
        <td id="LC937" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L938" class="blob-num js-line-number" data-line-number="938"></td>
        <td id="LC938" class="blob-code blob-code-inner js-file-line"><span class=pl-c>// }</span></td>
      </tr>
      <tr>
        <td id="L939" class="blob-num js-line-number" data-line-number="939"></td>
        <td id="LC939" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L940" class="blob-num js-line-number" data-line-number="940"></td>
        <td id="LC940" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L941" class="blob-num js-line-number" data-line-number="941"></td>
        <td id="LC941" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L942" class="blob-num js-line-number" data-line-number="942"></td>
        <td id="LC942" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Closed form solution for momentum iteration</span></td>
      </tr>
      <tr>
        <td id="L943" class="blob-num js-line-number" data-line-number="943"></td>
        <td id="LC943" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L944" class="blob-num js-line-number" data-line-number="944"></td>
        <td id="LC944" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  z_0 = zeros</span></td>
      </tr>
      <tr>
        <td id="L945" class="blob-num js-line-number" data-line-number="945"></td>
        <td id="LC945" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  w_0 = zeros</span></td>
      </tr>
      <tr>
        <td id="L946" class="blob-num js-line-number" data-line-number="946"></td>
        <td id="LC946" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L947" class="blob-num js-line-number" data-line-number="947"></td>
        <td id="LC947" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  z+ = beta*z + ([U*Lambda*U&#39;]*w - b)</span></td>
      </tr>
      <tr>
        <td id="L948" class="blob-num js-line-number" data-line-number="948"></td>
        <td id="LC948" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  w+ = w - alpha*z+</span></td>
      </tr>
      <tr>
        <td id="L949" class="blob-num js-line-number" data-line-number="949"></td>
        <td id="LC949" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L950" class="blob-num js-line-number" data-line-number="950"></td>
        <td id="LC950" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Usage</span></td>
      </tr>
      <tr>
        <td id="L951" class="blob-num js-line-number" data-line-number="951"></td>
        <td id="LC951" class="blob-code blob-code-inner js-file-line"><span class=pl-c></span></td>
      </tr>
      <tr>
        <td id="L952" class="blob-num js-line-number" data-line-number="952"></td>
        <td id="LC952" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  iter = getiter(U,Lambda, b, alpha)</span></td>
      </tr>
      <tr>
        <td id="L953" class="blob-num js-line-number" data-line-number="953"></td>
        <td id="LC953" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  w    = iter(1000) &lt;- gets the 1000&#39;th iteration.</span></td>
      </tr>
      <tr>
        <td id="L954" class="blob-num js-line-number" data-line-number="954"></td>
        <td id="LC954" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L955" class="blob-num js-line-number" data-line-number="955"></td>
        <td id="LC955" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>geniterMomentum</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>,</span> <span class=pl-v>Lambda</span><span class=pl-kos>,</span> <span class=pl-s1>b</span><span class=pl-kos>,</span> <span class=pl-s1>alpha</span><span class=pl-kos>,</span> <span class=pl-s1>beta</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L956" class="blob-num js-line-number" data-line-number="956"></td>
        <td id="LC956" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L957" class="blob-num js-line-number" data-line-number="957"></td>
        <td id="LC957" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Ub</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>mul</span><span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>,</span><span class=pl-s1>b</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L958" class="blob-num js-line-number" data-line-number="958"></td>
        <td id="LC958" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L959" class="blob-num js-line-number" data-line-number="959"></td>
        <td id="LC959" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Rmat</span> <span class=pl-c1>=</span> <span class=pl-k>function</span> <span class=pl-en>f</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L960" class="blob-num js-line-number" data-line-number="960"></td>
        <td id="LC960" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-kos>[</span> <span class=pl-s1>beta</span>          <span class=pl-kos>,</span> <span class=pl-v>Lambda</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span>          <span class=pl-kos>]</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L961" class="blob-num js-line-number" data-line-number="961"></td>
        <td id="LC961" class="blob-code blob-code-inner js-file-line">            <span class=pl-kos>[</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>*<span class=pl-s1>alpha</span>*<span class=pl-s1>beta</span> <span class=pl-kos>,</span> <span class=pl-c1>1</span> <span class=pl-c1>-</span> <span class=pl-s1>alpha</span>*<span class=pl-v>Lambda</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>]</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L962" class="blob-num js-line-number" data-line-number="962"></td>
        <td id="LC962" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L963" class="blob-num js-line-number" data-line-number="963"></td>
        <td id="LC963" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L964" class="blob-num js-line-number" data-line-number="964"></td>
        <td id="LC964" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>S</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>inv</span><span class=pl-kos>(</span> <span class=pl-kos>[</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-kos>[</span><span class=pl-s1>alpha</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L965" class="blob-num js-line-number" data-line-number="965"></td>
        <td id="LC965" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L966" class="blob-num js-line-number" data-line-number="966"></td>
        <td id="LC966" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fcoll</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L967" class="blob-num js-line-number" data-line-number="967"></td>
        <td id="LC967" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>maxLambda</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L968" class="blob-num js-line-number" data-line-number="968"></td>
        <td id="LC968" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>b</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L969" class="blob-num js-line-number" data-line-number="969"></td>
        <td id="LC969" class="blob-code blob-code-inner js-file-line">  	<span class=pl-s1>m</span> <span class=pl-c1>=</span> <span class=pl-en>matSum</span><span class=pl-kos>(</span><span class=pl-v>Rmat</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>S</span><span class=pl-kos>,</span><span class=pl-kos>[</span><span class=pl-v>Ub</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L970" class="blob-num js-line-number" data-line-number="970"></td>
        <td id="LC970" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fcoll</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>.</span><span class=pl-c1>matSum</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L971" class="blob-num js-line-number" data-line-number="971"></td>
        <td id="LC971" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>maxLambda</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>.</span><span class=pl-c1>lambda</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L972" class="blob-num js-line-number" data-line-number="972"></td>
        <td id="LC972" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L973" class="blob-num js-line-number" data-line-number="973"></td>
        <td id="LC973" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L974" class="blob-num js-line-number" data-line-number="974"></td>
        <td id="LC974" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-en>iter</span>: <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L975" class="blob-num js-line-number" data-line-number="975"></td>
        <td id="LC975" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>o</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L976" class="blob-num js-line-number" data-line-number="976"></td>
        <td id="LC976" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>b</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L977" class="blob-num js-line-number" data-line-number="977"></td>
        <td id="LC977" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>o</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>fcoll</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>(</span><span class=pl-s1>k</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L978" class="blob-num js-line-number" data-line-number="978"></td>
        <td id="LC978" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L979" class="blob-num js-line-number" data-line-number="979"></td>
        <td id="LC979" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-s1>o</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-v>U</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L980" class="blob-num js-line-number" data-line-number="980"></td>
        <td id="LC980" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>,</span> <span class=pl-c1>maxLambda</span>:<span class=pl-s1>maxLambda</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L981" class="blob-num js-line-number" data-line-number="981"></td>
        <td id="LC981" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L982" class="blob-num js-line-number" data-line-number="982"></td>
        <td id="LC982" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L983" class="blob-num js-line-number" data-line-number="983"></td>
        <td id="LC983" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Returns the path and coordinates of annotation for a circle-annotation */</span></td>
      </tr>
      <tr>
        <td id="L984" class="blob-num js-line-number" data-line-number="984"></td>
        <td id="LC984" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>ringPathGen</span><span class=pl-kos>(</span><span class=pl-s1>radius</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-kos>,</span> <span class=pl-s1>height</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L985" class="blob-num js-line-number" data-line-number="985"></td>
        <td id="LC985" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L986" class="blob-num js-line-number" data-line-number="986"></td>
        <td id="LC986" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>padding</span> <span class=pl-c1>=</span> <span class=pl-c1>4</span></td>
      </tr>
      <tr>
        <td id="L987" class="blob-num js-line-number" data-line-number="987"></td>
        <td id="LC987" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L988" class="blob-num js-line-number" data-line-number="988"></td>
        <td id="LC988" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>function</span> <span class=pl-en>ringPath</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L989" class="blob-num js-line-number" data-line-number="989"></td>
        <td id="LC989" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L990" class="blob-num js-line-number" data-line-number="990"></td>
        <td id="LC990" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// Generate Paths</span></td>
      </tr>
      <tr>
        <td id="L991" class="blob-num js-line-number" data-line-number="991"></td>
        <td id="LC991" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L992" class="blob-num js-line-number" data-line-number="992"></td>
        <td id="LC992" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L993" class="blob-num js-line-number" data-line-number="993"></td>
        <td id="LC993" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>xSign</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>x</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-kos>(</span><span class=pl-s1>x</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L994" class="blob-num js-line-number" data-line-number="994"></td>
        <td id="LC994" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>ySign</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L995" class="blob-num js-line-number" data-line-number="995"></td>
        <td id="LC995" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>r</span> <span class=pl-c1>=</span> <span class=pl-s1>radius</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L996" class="blob-num js-line-number" data-line-number="996"></td>
        <td id="LC996" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;&quot;</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L997" class="blob-num js-line-number" data-line-number="997"></td>
        <td id="LC997" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>a</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>r</span> * <span class=pl-s1>r</span> / <span class=pl-c1>2</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L998" class="blob-num js-line-number" data-line-number="998"></td>
        <td id="LC998" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>b</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-s1>r</span> * <span class=pl-s1>r</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-s1>y</span> * <span class=pl-s1>y</span><span class=pl-kos>,</span> <span class=pl-s1>x</span> * <span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L999" class="blob-num js-line-number" data-line-number="999"></td>
        <td id="LC999" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>c</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sqrt</span><span class=pl-kos>(</span><span class=pl-c1>2</span> * <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>min</span><span class=pl-kos>(</span><span class=pl-s1>x</span> * <span class=pl-s1>x</span><span class=pl-kos>,</span> <span class=pl-s1>y</span> * <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1000" class="blob-num js-line-number" data-line-number="1000"></td>
        <td id="LC1000" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dir</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L1001" class="blob-num js-line-number" data-line-number="1001"></td>
        <td id="LC1001" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>x</span> * <span class=pl-s1>x</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span> * <span class=pl-s1>y</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>r</span> * <span class=pl-s1>r</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1002" class="blob-num js-line-number" data-line-number="1002"></td>
        <td id="LC1002" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;&quot;</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1003" class="blob-num js-line-number" data-line-number="1003"></td>
        <td id="LC1003" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>c</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>r</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1004" class="blob-num js-line-number" data-line-number="1004"></td>
        <td id="LC1004" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>&gt;</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1005" class="blob-num js-line-number" data-line-number="1005"></td>
        <td id="LC1005" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dir</span> <span class=pl-c1>=</span> <span class=pl-s1>xSign</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span> ? <span class=pl-s>&quot;E&quot;</span> : <span class=pl-s>&quot;W&quot;</span></td>
      </tr>
      <tr>
        <td id="L1006" class="blob-num js-line-number" data-line-number="1006"></td>
        <td id="LC1006" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;M&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>xSign</span> * <span class=pl-s1>b</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1007" class="blob-num js-line-number" data-line-number="1007"></td>
        <td id="LC1007" class="blob-code blob-code-inner js-file-line">          <span class=pl-c1>+</span> <span class=pl-s>&quot;,L&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1008" class="blob-num js-line-number" data-line-number="1008"></td>
        <td id="LC1008" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1009" class="blob-num js-line-number" data-line-number="1009"></td>
        <td id="LC1009" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dir</span> <span class=pl-c1>=</span> <span class=pl-s1>ySign</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span> ? <span class=pl-s>&quot;S&quot;</span> : <span class=pl-s>&quot;N&quot;</span></td>
      </tr>
      <tr>
        <td id="L1010" class="blob-num js-line-number" data-line-number="1010"></td>
        <td id="LC1010" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;M&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>ySign</span> * <span class=pl-s1>b</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1011" class="blob-num js-line-number" data-line-number="1011"></td>
        <td id="LC1011" class="blob-code blob-code-inner js-file-line">         <span class=pl-c1>+</span> <span class=pl-s>&quot;,L&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1012" class="blob-num js-line-number" data-line-number="1012"></td>
        <td id="LC1012" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1013" class="blob-num js-line-number" data-line-number="1013"></td>
        <td id="LC1013" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>&gt;</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1014" class="blob-num js-line-number" data-line-number="1014"></td>
        <td id="LC1014" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>dir</span> <span class=pl-c1>=</span> <span class=pl-s1>xSign</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span> ? <span class=pl-s>&quot;E&quot;</span> : <span class=pl-s>&quot;W&quot;</span></td>
      </tr>
      <tr>
        <td id="L1015" class="blob-num js-line-number" data-line-number="1015"></td>
        <td id="LC1015" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;M&quot;</span>  <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>xSign</span> * <span class=pl-s1>a</span><span class=pl-kos>)</span>           <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>ySign</span> * <span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-c1>+</span></td>
      </tr>
      <tr>
        <td id="L1016" class="blob-num js-line-number" data-line-number="1016"></td>
        <td id="LC1016" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>&quot;,L&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>xSign</span> * <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span> <span class=pl-c1>+</span></td>
      </tr>
      <tr>
        <td id="L1017" class="blob-num js-line-number" data-line-number="1017"></td>
        <td id="LC1017" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>&quot;L&quot;</span>  <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span>                   <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1018" class="blob-num js-line-number" data-line-number="1018"></td>
        <td id="LC1018" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span> <span class=pl-k>else</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1019" class="blob-num js-line-number" data-line-number="1019"></td>
        <td id="LC1019" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>dir</span> <span class=pl-c1>=</span> <span class=pl-s1>ySign</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span> ? <span class=pl-s>&quot;S&quot;</span> : <span class=pl-s>&quot;N&quot;</span></td>
      </tr>
      <tr>
        <td id="L1020" class="blob-num js-line-number" data-line-number="1020"></td>
        <td id="LC1020" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s>&quot;M&quot;</span>  <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>xSign</span> * <span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>ySign</span> * <span class=pl-s1>a</span><span class=pl-kos>)</span> <span class=pl-c1>+</span></td>
      </tr>
      <tr>
        <td id="L1021" class="blob-num js-line-number" data-line-number="1021"></td>
        <td id="LC1021" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>&quot;,L&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span>         <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>ySign</span> * <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-c1>+</span></td>
      </tr>
      <tr>
        <td id="L1022" class="blob-num js-line-number" data-line-number="1022"></td>
        <td id="LC1022" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>&quot;L&quot;</span>  <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>x</span><span class=pl-kos>)</span>         <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1023" class="blob-num js-line-number" data-line-number="1023"></td>
        <td id="LC1023" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1024" class="blob-num js-line-number" data-line-number="1024"></td>
        <td id="LC1024" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1025" class="blob-num js-line-number" data-line-number="1025"></td>
        <td id="LC1025" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>top</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L1026" class="blob-num js-line-number" data-line-number="1026"></td>
        <td id="LC1026" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>left</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L1027" class="blob-num js-line-number" data-line-number="1027"></td>
        <td id="LC1027" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1028" class="blob-num js-line-number" data-line-number="1028"></td>
        <td id="LC1028" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// Generate Paths</span></td>
      </tr>
      <tr>
        <td id="L1029" class="blob-num js-line-number" data-line-number="1029"></td>
        <td id="LC1029" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// if (dir == &quot;S&quot;) { top  = (p2[1] + padding); left = (p2[0] - width/2) }</span></td>
      </tr>
      <tr>
        <td id="L1030" class="blob-num js-line-number" data-line-number="1030"></td>
        <td id="LC1030" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// if (dir == &quot;N&quot;) { top  = (p2[1] - height - padding); left = (p2[0] - width/2) }</span></td>
      </tr>
      <tr>
        <td id="L1031" class="blob-num js-line-number" data-line-number="1031"></td>
        <td id="LC1031" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// if (dir == &quot;W&quot;) { top  = (p2[1] - height/2); left = (p2[0] - width - padding) }</span></td>
      </tr>
      <tr>
        <td id="L1032" class="blob-num js-line-number" data-line-number="1032"></td>
        <td id="LC1032" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>// if (dir == &quot;E&quot;) { top  = (p2[1] - height/2); left = (p2[0] + padding) }</span></td>
      </tr>
      <tr>
        <td id="L1033" class="blob-num js-line-number" data-line-number="1033"></td>
        <td id="LC1033" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1034" class="blob-num js-line-number" data-line-number="1034"></td>
        <td id="LC1034" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>dir</span> <span class=pl-c1>==</span> <span class=pl-s>&quot;S&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>top</span>  <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>height</span>/<span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-s1>padding</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>left</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>width</span>/<span class=pl-c1>2</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1035" class="blob-num js-line-number" data-line-number="1035"></td>
        <td id="LC1035" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>dir</span> <span class=pl-c1>==</span> <span class=pl-s>&quot;N&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>top</span>  <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>padding</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>left</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>width</span>/<span class=pl-c1>2</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1036" class="blob-num js-line-number" data-line-number="1036"></td>
        <td id="LC1036" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>dir</span> <span class=pl-c1>==</span> <span class=pl-s>&quot;W&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>top</span>  <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>height</span>/<span class=pl-c1>4</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>left</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>width</span> <span class=pl-c1>-</span> <span class=pl-s1>padding</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1037" class="blob-num js-line-number" data-line-number="1037"></td>
        <td id="LC1037" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>dir</span> <span class=pl-c1>==</span> <span class=pl-s>&quot;E&quot;</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>top</span>  <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>height</span>/<span class=pl-c1>4</span><span class=pl-kos>)</span><span class=pl-kos>;</span> <span class=pl-s1>left</span> <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s1>padding</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1038" class="blob-num js-line-number" data-line-number="1038"></td>
        <td id="LC1038" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1039" class="blob-num js-line-number" data-line-number="1039"></td>
        <td id="LC1039" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-c1>d</span>:<span class=pl-s1>d</span><span class=pl-kos>,</span> <span class=pl-c1>label</span>:<span class=pl-kos>[</span><span class=pl-s1>left</span><span class=pl-kos>,</span> <span class=pl-s1>top</span><span class=pl-kos>]</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1040" class="blob-num js-line-number" data-line-number="1040"></td>
        <td id="LC1040" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1041" class="blob-num js-line-number" data-line-number="1041"></td>
        <td id="LC1041" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1042" class="blob-num js-line-number" data-line-number="1042"></td>
        <td id="LC1042" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1043" class="blob-num js-line-number" data-line-number="1043"></td>
        <td id="LC1043" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>ringPath</span></td>
      </tr>
      <tr>
        <td id="L1044" class="blob-num js-line-number" data-line-number="1044"></td>
        <td id="LC1044" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1045" class="blob-num js-line-number" data-line-number="1045"></td>
        <td id="LC1045" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1046" class="blob-num js-line-number" data-line-number="1046"></td>
        <td id="LC1046" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1047" class="blob-num js-line-number" data-line-number="1047"></td>
        <td id="LC1047" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>colorMap</span><span class=pl-kos>(</span><span class=pl-s1>root</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-kos>,</span> <span class=pl-s1>colorScale</span><span class=pl-kos>,</span> <span class=pl-s1>axisScale</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1048" class="blob-num js-line-number" data-line-number="1048"></td>
        <td id="LC1048" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1049" class="blob-num js-line-number" data-line-number="1049"></td>
        <td id="LC1049" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>margin</span> <span class=pl-c1>=</span> <span class=pl-kos>{</span> <span class=pl-c1>top</span>: <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>right</span>: <span class=pl-c1>12</span><span class=pl-kos>,</span> <span class=pl-c1>bottom</span>: <span class=pl-c1>30</span><span class=pl-kos>,</span> <span class=pl-c1>left</span>: <span class=pl-c1>12</span> <span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1050" class="blob-num js-line-number" data-line-number="1050"></td>
        <td id="LC1050" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>height</span> <span class=pl-c1>=</span> <span class=pl-c1>12</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1051" class="blob-num js-line-number" data-line-number="1051"></td>
        <td id="LC1051" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1052" class="blob-num js-line-number" data-line-number="1052"></td>
        <td id="LC1052" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>root</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-kos>(</span><span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span><span class=pl-kos>)</span>  <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1053" class="blob-num js-line-number" data-line-number="1053"></td>
        <td id="LC1053" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>root</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-kos>(</span><span class=pl-s1>height</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>top</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>bottom</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1054" class="blob-num js-line-number" data-line-number="1054"></td>
        <td id="LC1054" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>canvas</span> <span class=pl-c1>=</span> <span class=pl-s1>root</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;canvas&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1055" class="blob-num js-line-number" data-line-number="1055"></td>
        <td id="LC1055" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-c1>+</span><span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1056" class="blob-num js-line-number" data-line-number="1056"></td>
        <td id="LC1056" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>height</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1057" class="blob-num js-line-number" data-line-number="1057"></td>
        <td id="LC1057" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;relative&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1058" class="blob-num js-line-number" data-line-number="1058"></td>
        <td id="LC1058" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;left&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1059" class="blob-num js-line-number" data-line-number="1059"></td>
        <td id="LC1059" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;top&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;8px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1060" class="blob-num js-line-number" data-line-number="1060"></td>
        <td id="LC1060" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>svg</span> <span class=pl-c1>=</span> root.<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1061" class="blob-num js-line-number" data-line-number="1061"></td>
        <td id="LC1061" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>right</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1062" class="blob-num js-line-number" data-line-number="1062"></td>
        <td id="LC1062" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>40</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1063" class="blob-num js-line-number" data-line-number="1063"></td>
        <td id="LC1063" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;left&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>-</span><span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1064" class="blob-num js-line-number" data-line-number="1064"></td>
        <td id="LC1064" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.5</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1065" class="blob-num js-line-number" data-line-number="1065"></td>
        <td id="LC1065" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1066" class="blob-num js-line-number" data-line-number="1066"></td>
        <td id="LC1066" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>margin</span><span class=pl-kos>.</span><span class=pl-c1>left</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;, 0)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1067" class="blob-num js-line-number" data-line-number="1067"></td>
        <td id="LC1067" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;figtext&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1068" class="blob-num js-line-number" data-line-number="1068"></td>
        <td id="LC1068" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>axis</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>axisBottom</span><span class=pl-kos>(</span><span class=pl-s1>axisScale</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>ticks</span><span class=pl-kos>(</span><span class=pl-c1>5</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1069" class="blob-num js-line-number" data-line-number="1069"></td>
        <td id="LC1069" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>axis</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1070" class="blob-num js-line-number" data-line-number="1070"></td>
        <td id="LC1070" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>root</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-s>&quot;.label&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span><span class=pl-s>&quot;Activation value&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1071" class="blob-num js-line-number" data-line-number="1071"></td>
        <td id="LC1071" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>context</span> <span class=pl-c1>=</span> <span class=pl-s1>canvas</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>getContext</span><span class=pl-kos>(</span><span class=pl-s>&quot;2d&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1072" class="blob-num js-line-number" data-line-number="1072"></td>
        <td id="LC1072" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>width</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1073" class="blob-num js-line-number" data-line-number="1073"></td>
        <td id="LC1073" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>context</span><span class=pl-kos>.</span><span class=pl-c1>fillStyle</span> <span class=pl-c1>=</span> <span class=pl-s1>colorScale</span><span class=pl-kos>(</span><span class=pl-s1>axisScale</span><span class=pl-kos>.</span><span class=pl-en>invert</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1074" class="blob-num js-line-number" data-line-number="1074"></td>
        <td id="LC1074" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>context</span><span class=pl-kos>.</span><span class=pl-en>fillRect</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>,</span> <span class=pl-s1>height</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1075" class="blob-num js-line-number" data-line-number="1075"></td>
        <td id="LC1075" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1076" class="blob-num js-line-number" data-line-number="1076"></td>
        <td id="LC1076" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1077" class="blob-num js-line-number" data-line-number="1077"></td>
        <td id="LC1077" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1078" class="blob-num js-line-number" data-line-number="1078"></td>
        <td id="LC1078" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1079" class="blob-num js-line-number" data-line-number="1079"></td>
        <td id="LC1079" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>sliderBarGen</span><span class=pl-kos>(</span><span class=pl-s1>barlengths</span><span class=pl-kos>,</span> <span class=pl-s1>expfn</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1080" class="blob-num js-line-number" data-line-number="1080"></td>
        <td id="LC1080" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1081" class="blob-num js-line-number" data-line-number="1081"></td>
        <td id="LC1081" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-en>update</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1082" class="blob-num js-line-number" data-line-number="1082"></td>
        <td id="LC1082" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>height</span> <span class=pl-c1>=</span> <span class=pl-c1>60</span></td>
      </tr>
      <tr>
        <td id="L1083" class="blob-num js-line-number" data-line-number="1083"></td>
        <td id="LC1083" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>maxX</span> <span class=pl-c1>=</span> <span class=pl-c1>14.2</span></td>
      </tr>
      <tr>
        <td id="L1084" class="blob-num js-line-number" data-line-number="1084"></td>
        <td id="LC1084" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>modval</span> <span class=pl-c1>=</span> <span class=pl-c1>6</span></td>
      </tr>
      <tr>
        <td id="L1085" class="blob-num js-line-number" data-line-number="1085"></td>
        <td id="LC1085" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>strokewidth</span> <span class=pl-c1>=</span> <span class=pl-c1>2</span></td>
      </tr>
      <tr>
        <td id="L1086" class="blob-num js-line-number" data-line-number="1086"></td>
        <td id="LC1086" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>gap</span> <span class=pl-c1>=</span> <span class=pl-c1>15</span></td>
      </tr>
      <tr>
        <td id="L1087" class="blob-num js-line-number" data-line-number="1087"></td>
        <td id="LC1087" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-en>mouseover</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1088" class="blob-num js-line-number" data-line-number="1088"></td>
        <td id="LC1088" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-en>labelFunc</span> <span class=pl-c1>=</span> <span class=pl-k>function</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> ? <span class=pl-s>&quot;Eigenvalue 1&quot;</span> : <span class=pl-s>&quot;&quot;</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-kos>(</span> <span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-c1>+</span><span class=pl-c1>1</span><span class=pl-kos>)</span> % <span class=pl-s1>modval</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span> <span class=pl-kos>)</span> ? <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>)</span> : <span class=pl-s>&quot;&quot;</span><span class=pl-kos>)</span>  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1089" class="blob-num js-line-number" data-line-number="1089"></td>
        <td id="LC1089" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1090" class="blob-num js-line-number" data-line-number="1090"></td>
        <td id="LC1090" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>function</span> <span class=pl-en>sliderBar</span><span class=pl-kos>(</span><span class=pl-s1>div</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1091" class="blob-num js-line-number" data-line-number="1091"></td>
        <td id="LC1091" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1092" class="blob-num js-line-number" data-line-number="1092"></td>
        <td id="LC1092" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>slider</span> <span class=pl-c1>=</span> <span class=pl-s1>div</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;div&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1093" class="blob-num js-line-number" data-line-number="1093"></td>
        <td id="LC1093" class="blob-code blob-code-inner js-file-line">	                 <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;relative&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1094" class="blob-num js-line-number" data-line-number="1094"></td>
        <td id="LC1094" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1095" class="blob-num js-line-number" data-line-number="1095"></td>
        <td id="LC1095" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-en>updateEverything</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>,</span> <span class=pl-s1>circ</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1096" class="blob-num js-line-number" data-line-number="1096"></td>
        <td id="LC1096" class="blob-code blob-code-inner js-file-line">	      <span class=pl-s1>step</span><span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-s>&quot;Step k = &quot;</span> <span class=pl-c1>+</span> <span class=pl-en>numberWithCommas</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>floor</span><span class=pl-kos>(</span><span class=pl-s1>expfn</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1097" class="blob-num js-line-number" data-line-number="1097"></td>
        <td id="LC1097" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1098" class="blob-num js-line-number" data-line-number="1098"></td>
        <td id="LC1098" class="blob-code blob-code-inner js-file-line">	      <span class=pl-k>if</span> <span class=pl-kos>(</span>!<span class=pl-kos>(</span><span class=pl-s1>circ</span> <span class=pl-c1>===</span> undefined<span class=pl-kos>)</span> <span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1099" class="blob-num js-line-number" data-line-number="1099"></td>
        <td id="LC1099" class="blob-code blob-code-inner js-file-line">	        <span class=pl-k>var</span> <span class=pl-s1>ctm</span> <span class=pl-c1>=</span> <span class=pl-s1>circ</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>getCTM</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1100" class="blob-num js-line-number" data-line-number="1100"></td>
        <td id="LC1100" class="blob-code blob-code-inner js-file-line">	        <span class=pl-en>setTM</span><span class=pl-kos>(</span><span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-s1>ctm</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1101" class="blob-num js-line-number" data-line-number="1101"></td>
        <td id="LC1101" class="blob-code blob-code-inner js-file-line">	        <span class=pl-k>var</span> <span class=pl-s1>barnodes</span> <span class=pl-c1>=</span> <span class=pl-s1>bars</span><span class=pl-kos>.</span><span class=pl-en>nodes</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1102" class="blob-num js-line-number" data-line-number="1102"></td>
        <td id="LC1102" class="blob-code blob-code-inner js-file-line">	        <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>barlengths</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>j</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1103" class="blob-num js-line-number" data-line-number="1103"></td>
        <td id="LC1103" class="blob-code blob-code-inner js-file-line">	          <span class=pl-k>var</span> <span class=pl-s1>r</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>barlengths</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-c1>-</span><span class=pl-c1>0.01</span><span class=pl-kos>,</span><span class=pl-s1>barlengths</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-c1>-</span><span class=pl-c1>0.01</span><span class=pl-kos>,</span><span class=pl-s1>barlengths</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-c1>1</span>/<span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-c1>0.2</span><span class=pl-kos>,</span> <span class=pl-c1>0.2</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1104" class="blob-num js-line-number" data-line-number="1104"></td>
        <td id="LC1104" class="blob-code blob-code-inner js-file-line">	          <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-s1>barnodes</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span><span class=pl-s1>r</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1105" class="blob-num js-line-number" data-line-number="1105"></td>
        <td id="LC1105" class="blob-code blob-code-inner js-file-line">	          <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>i</span> <span class=pl-c1>&gt;</span> <span class=pl-s1>barlengths</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1106" class="blob-num js-line-number" data-line-number="1106"></td>
        <td id="LC1106" class="blob-code blob-code-inner js-file-line">	             <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-s1>barnodes</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1107" class="blob-num js-line-number" data-line-number="1107"></td>
        <td id="LC1107" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>}</span> <span class=pl-k>else</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1108" class="blob-num js-line-number" data-line-number="1108"></td>
        <td id="LC1108" class="blob-code blob-code-inner js-file-line">	             <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-s1>barnodes</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1109" class="blob-num js-line-number" data-line-number="1109"></td>
        <td id="LC1109" class="blob-code blob-code-inner js-file-line">	          <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1110" class="blob-num js-line-number" data-line-number="1110"></td>
        <td id="LC1110" class="blob-code blob-code-inner js-file-line">	        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1111" class="blob-num js-line-number" data-line-number="1111"></td>
        <td id="LC1111" class="blob-code blob-code-inner js-file-line">	      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1112" class="blob-num js-line-number" data-line-number="1112"></td>
        <td id="LC1112" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1113" class="blob-num js-line-number" data-line-number="1113"></td>
        <td id="LC1113" class="blob-code blob-code-inner js-file-line">	      <span class=pl-en>update</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1114" class="blob-num js-line-number" data-line-number="1114"></td>
        <td id="LC1114" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1115" class="blob-num js-line-number" data-line-number="1115"></td>
        <td id="LC1115" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1116" class="blob-num js-line-number" data-line-number="1116"></td>
        <td id="LC1116" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>slidera</span> <span class=pl-c1>=</span> <span class=pl-en>sliderGen</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>940</span><span class=pl-kos>,</span> <span class=pl-c1>60</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1117" class="blob-num js-line-number" data-line-number="1117"></td>
        <td id="LC1117" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>ticks</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span> <span class=pl-s1>maxX</span>/<span class=pl-c1>5</span><span class=pl-kos>,</span> <span class=pl-c1>2</span>*<span class=pl-s1>maxX</span>/<span class=pl-c1>5</span><span class=pl-kos>,</span> <span class=pl-c1>3</span>*<span class=pl-s1>maxX</span>/<span class=pl-c1>5</span><span class=pl-kos>,</span> <span class=pl-c1>4</span>*<span class=pl-s1>maxX</span>/<span class=pl-c1>5</span><span class=pl-kos>,</span> <span class=pl-s1>maxX</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1118" class="blob-num js-line-number" data-line-number="1118"></td>
        <td id="LC1118" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>ticktitles</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-en>numberWithCommas</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>floor</span><span class=pl-kos>(</span><span class=pl-s1>expfn</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1119" class="blob-num js-line-number" data-line-number="1119"></td>
        <td id="LC1119" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>cRadius</span><span class=pl-kos>(</span><span class=pl-c1>7</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1120" class="blob-num js-line-number" data-line-number="1120"></td>
        <td id="LC1120" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>startxval</span><span class=pl-kos>(</span><span class=pl-c1>4</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1121" class="blob-num js-line-number" data-line-number="1121"></td>
        <td id="LC1121" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>shifty</span><span class=pl-kos>(</span><span class=pl-c1>3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1122" class="blob-num js-line-number" data-line-number="1122"></td>
        <td id="LC1122" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>margin</span><span class=pl-kos>(</span><span class=pl-kos>{</span><span class=pl-c1>right</span>: <span class=pl-c1>160</span><span class=pl-kos>,</span> <span class=pl-c1>left</span>: <span class=pl-c1>140</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1123" class="blob-num js-line-number" data-line-number="1123"></td>
        <td id="LC1123" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>.</span><span class=pl-en>change</span><span class=pl-kos>(</span><span class=pl-en>updateEverything</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1124" class="blob-num js-line-number" data-line-number="1124"></td>
        <td id="LC1124" class="blob-code blob-code-inner js-file-line">	    <span class=pl-kos>(</span><span class=pl-s1>slider</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1125" class="blob-num js-line-number" data-line-number="1125"></td>
        <td id="LC1125" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1126" class="blob-num js-line-number" data-line-number="1126"></td>
        <td id="LC1126" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>width</span>  <span class=pl-c1>=</span> <span class=pl-c1>695</span><span class=pl-c1>+</span><span class=pl-c1>80</span></td>
      </tr>
      <tr>
        <td id="L1127" class="blob-num js-line-number" data-line-number="1127"></td>
        <td id="LC1127" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>svg</span> <span class=pl-c1>=</span> <span class=pl-s1>slider</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1128" class="blob-num js-line-number" data-line-number="1128"></td>
        <td id="LC1128" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1129" class="blob-num js-line-number" data-line-number="1129"></td>
        <td id="LC1129" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>step</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;figtext&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span><span class=pl-c1>145</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span><span class=pl-c1>15</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-s>&quot;Step k = &quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1130" class="blob-num js-line-number" data-line-number="1130"></td>
        <td id="LC1130" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1131" class="blob-num js-line-number" data-line-number="1131"></td>
        <td id="LC1131" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>maxX</span><span class=pl-kos>,</span> <span class=pl-c1>100000</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>90</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-c1>-</span><span class=pl-c1>45</span><span class=pl-kos>,</span><span class=pl-s1>width</span><span class=pl-c1>+</span><span class=pl-c1>45</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1132" class="blob-num js-line-number" data-line-number="1132"></td>
        <td id="LC1132" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>scaleLinear</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>domain</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>,</span><span class=pl-s1>barlengths</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>range</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>10</span><span class=pl-kos>,</span> <span class=pl-s1>height</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1133" class="blob-num js-line-number" data-line-number="1133"></td>
        <td id="LC1133" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1134" class="blob-num js-line-number" data-line-number="1134"></td>
        <td id="LC1134" class="blob-code blob-code-inner js-file-line">	<span class=pl-c>//  line.moveToBack()</span></td>
      </tr>
      <tr>
        <td id="L1135" class="blob-num js-line-number" data-line-number="1135"></td>
        <td id="LC1135" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1136" class="blob-num js-line-number" data-line-number="1136"></td>
        <td id="LC1136" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>line</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1137" class="blob-num js-line-number" data-line-number="1137"></td>
        <td id="LC1137" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1138" class="blob-num js-line-number" data-line-number="1138"></td>
        <td id="LC1138" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1139" class="blob-num js-line-number" data-line-number="1139"></td>
        <td id="LC1139" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1140" class="blob-num js-line-number" data-line-number="1140"></td>
        <td id="LC1140" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>height</span><span class=pl-c1>+</span><span class=pl-c1>50</span><span class=pl-c1>+</span><span class=pl-s1>gap</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1141" class="blob-num js-line-number" data-line-number="1141"></td>
        <td id="LC1141" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1142" class="blob-num js-line-number" data-line-number="1142"></td>
        <td id="LC1142" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;1px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1143" class="blob-num js-line-number" data-line-number="1143"></td>
        <td id="LC1143" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1144" class="blob-num js-line-number" data-line-number="1144"></td>
        <td id="LC1144" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>moveToBack</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1145" class="blob-num js-line-number" data-line-number="1145"></td>
        <td id="LC1145" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>moveToFront</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1146" class="blob-num js-line-number" data-line-number="1146"></td>
        <td id="LC1146" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1147" class="blob-num js-line-number" data-line-number="1147"></td>
        <td id="LC1147" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>chart</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>940</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1148" class="blob-num js-line-number" data-line-number="1148"></td>
        <td id="LC1148" class="blob-code blob-code-inner js-file-line">	                 <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-kos>(</span><span class=pl-s1>height</span><span class=pl-c1>+</span><span class=pl-c1>100</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1149" class="blob-num js-line-number" data-line-number="1149"></td>
        <td id="LC1149" class="blob-code blob-code-inner js-file-line">	                 <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;top&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;30px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1150" class="blob-num js-line-number" data-line-number="1150"></td>
        <td id="LC1150" class="blob-code blob-code-inner js-file-line">	                 <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1151" class="blob-num js-line-number" data-line-number="1151"></td>
        <td id="LC1151" class="blob-code blob-code-inner js-file-line">	                 <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(50, &quot;</span> <span class=pl-c1>+</span> <span class=pl-kos>(</span><span class=pl-s1>gap</span><span class=pl-c1>+</span><span class=pl-c1>60</span><span class=pl-kos>)</span> <span class=pl-c1>+</span><span class=pl-s>&quot; )&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1152" class="blob-num js-line-number" data-line-number="1152"></td>
        <td id="LC1152" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1153" class="blob-num js-line-number" data-line-number="1153"></td>
        <td id="LC1153" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1154" class="blob-num js-line-number" data-line-number="1154"></td>
        <td id="LC1154" class="blob-code blob-code-inner js-file-line">	  chart.selectAll(&quot;rect&quot;).data(barlengths)</td>
      </tr>
      <tr>
        <td id="L1155" class="blob-num js-line-number" data-line-number="1155"></td>
        <td id="LC1155" class="blob-code blob-code-inner js-file-line">	     .enter()</td>
      </tr>
      <tr>
        <td id="L1156" class="blob-num js-line-number" data-line-number="1156"></td>
        <td id="LC1156" class="blob-code blob-code-inner js-file-line">	     .append(&quot;rect&quot;)</td>
      </tr>
      <tr>
        <td id="L1157" class="blob-num js-line-number" data-line-number="1157"></td>
        <td id="LC1157" class="blob-code blob-code-inner js-file-line">	     .<span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1158" class="blob-num js-line-number" data-line-number="1158"></td>
        <td id="LC1158" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-en>y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-c1>-</span><span class=pl-c1>2</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1159" class="blob-num js-line-number" data-line-number="1159"></td>
        <td id="LC1159" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>maxX</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-c1>90</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1160" class="blob-num js-line-number" data-line-number="1160"></td>
        <td id="LC1160" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>4</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1161" class="blob-num js-line-number" data-line-number="1161"></td>
        <td id="LC1161" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;opacity&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0.01</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1162" class="blob-num js-line-number" data-line-number="1162"></td>
        <td id="LC1162" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;gray&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1163" class="blob-num js-line-number" data-line-number="1163"></td>
        <td id="LC1163" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;mouseover&quot;</span><span class=pl-kos>,</span> <span class=pl-en>mouseover</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1164" class="blob-num js-line-number" data-line-number="1164"></td>
        <td id="LC1164" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;mouseout&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-en>update</span><span class=pl-kos>(</span><span class=pl-s1>slidera</span><span class=pl-kos>.</span><span class=pl-en>xval</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1165" class="blob-num js-line-number" data-line-number="1165"></td>
        <td id="LC1165" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1166" class="blob-num js-line-number" data-line-number="1166"></td>
        <td id="LC1166" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>var</span> <span class=pl-s1>bars</span> <span class=pl-c1>=</span> chart.selectAll(&quot;line&quot;).<span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>barlengths</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1167" class="blob-num js-line-number" data-line-number="1167"></td>
        <td id="LC1167" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>enter</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1168" class="blob-num js-line-number" data-line-number="1168"></td>
        <td id="LC1168" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1169" class="blob-num js-line-number" data-line-number="1169"></td>
        <td id="LC1169" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x1&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1170" class="blob-num js-line-number" data-line-number="1170"></td>
        <td id="LC1170" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y1&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1171" class="blob-num js-line-number" data-line-number="1171"></td>
        <td id="LC1171" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1172" class="blob-num js-line-number" data-line-number="1172"></td>
        <td id="LC1172" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1173" class="blob-num js-line-number" data-line-number="1173"></td>
        <td id="LC1173" class="blob-code blob-code-inner js-file-line">	     <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>strokewidth</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1174" class="blob-num js-line-number" data-line-number="1174"></td>
        <td id="LC1174" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1175" class="blob-num js-line-number" data-line-number="1175"></td>
        <td id="LC1175" class="blob-code blob-code-inner js-file-line">	  chart.selectAll(&quot;text&quot;).data(barlengths)</td>
      </tr>
      <tr>
        <td id="L1176" class="blob-num js-line-number" data-line-number="1176"></td>
        <td id="LC1176" class="blob-code blob-code-inner js-file-line">	       .enter()</td>
      </tr>
      <tr>
        <td id="L1177" class="blob-num js-line-number" data-line-number="1177"></td>
        <td id="LC1177" class="blob-code blob-code-inner js-file-line">	       .<span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;text&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1178" class="blob-num js-line-number" data-line-number="1178"></td>
        <td id="LC1178" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;figtext2&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1179" class="blob-num js-line-number" data-line-number="1179"></td>
        <td id="LC1179" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;text-anchor&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;end&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1180" class="blob-num js-line-number" data-line-number="1180"></td>
        <td id="LC1180" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>75</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1181" class="blob-num js-line-number" data-line-number="1181"></td>
        <td id="LC1181" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>y</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-c1>4</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1182" class="blob-num js-line-number" data-line-number="1182"></td>
        <td id="LC1182" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>150</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1183" class="blob-num js-line-number" data-line-number="1183"></td>
        <td id="LC1183" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;gray&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1184" class="blob-num js-line-number" data-line-number="1184"></td>
        <td id="LC1184" class="blob-code blob-code-inner js-file-line">	       <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-en>labelFunc</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1185" class="blob-num js-line-number" data-line-number="1185"></td>
        <td id="LC1185" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1186" class="blob-num js-line-number" data-line-number="1186"></td>
        <td id="LC1186" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>chart</span><span class=pl-kos>.</span><span class=pl-en>moveToBack</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1187" class="blob-num js-line-number" data-line-number="1187"></td>
        <td id="LC1187" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1188" class="blob-num js-line-number" data-line-number="1188"></td>
        <td id="LC1188" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>function</span> <span class=pl-en>updateBars</span><span class=pl-kos>(</span><span class=pl-s1>barlengths_in</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1189" class="blob-num js-line-number" data-line-number="1189"></td>
        <td id="LC1189" class="blob-code blob-code-inner js-file-line">	  	<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>barlengths</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>barlengths</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>barlengths_in</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1190" class="blob-num js-line-number" data-line-number="1190"></td>
        <td id="LC1190" class="blob-code blob-code-inner js-file-line">	  	<span class=pl-s1>chart</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&quot;line&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>data</span><span class=pl-kos>(</span><span class=pl-s1>barlengths_in</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>merge</span><span class=pl-kos>(</span><span class=pl-s1>chart</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x2&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1191" class="blob-num js-line-number" data-line-number="1191"></td>
        <td id="LC1191" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1192" class="blob-num js-line-number" data-line-number="1192"></td>
        <td id="LC1192" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1193" class="blob-num js-line-number" data-line-number="1193"></td>
        <td id="LC1193" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-c1>slidera</span>:<span class=pl-s1>slidera</span><span class=pl-kos>,</span> <span class=pl-c1>update</span>:<span class=pl-s1>updateBars</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1194" class="blob-num js-line-number" data-line-number="1194"></td>
        <td id="LC1194" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1195" class="blob-num js-line-number" data-line-number="1195"></td>
        <td id="LC1195" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1196" class="blob-num js-line-number" data-line-number="1196"></td>
        <td id="LC1196" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>height</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1197" class="blob-num js-line-number" data-line-number="1197"></td>
        <td id="LC1197" class="blob-code blob-code-inner js-file-line">		<span class=pl-s1>height</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1198" class="blob-num js-line-number" data-line-number="1198"></td>
        <td id="LC1198" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1199" class="blob-num js-line-number" data-line-number="1199"></td>
        <td id="LC1199" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1200" class="blob-num js-line-number" data-line-number="1200"></td>
        <td id="LC1200" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>linewidth</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1201" class="blob-num js-line-number" data-line-number="1201"></td>
        <td id="LC1201" class="blob-code blob-code-inner js-file-line">		<span class=pl-s1>strokewidth</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1202" class="blob-num js-line-number" data-line-number="1202"></td>
        <td id="LC1202" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1203" class="blob-num js-line-number" data-line-number="1203"></td>
        <td id="LC1203" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1204" class="blob-num js-line-number" data-line-number="1204"></td>
        <td id="LC1204" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>maxX</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1205" class="blob-num js-line-number" data-line-number="1205"></td>
        <td id="LC1205" class="blob-code blob-code-inner js-file-line">		<span class=pl-s1>maxX</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1206" class="blob-num js-line-number" data-line-number="1206"></td>
        <td id="LC1206" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1207" class="blob-num js-line-number" data-line-number="1207"></td>
        <td id="LC1207" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1208" class="blob-num js-line-number" data-line-number="1208"></td>
        <td id="LC1208" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>update</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1209" class="blob-num js-line-number" data-line-number="1209"></td>
        <td id="LC1209" class="blob-code blob-code-inner js-file-line">		<span class=pl-en>update</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1210" class="blob-num js-line-number" data-line-number="1210"></td>
        <td id="LC1210" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1211" class="blob-num js-line-number" data-line-number="1211"></td>
        <td id="LC1211" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1212" class="blob-num js-line-number" data-line-number="1212"></td>
        <td id="LC1212" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>mouseover</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1213" class="blob-num js-line-number" data-line-number="1213"></td>
        <td id="LC1213" class="blob-code blob-code-inner js-file-line">		<span class=pl-en>mouseover</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1214" class="blob-num js-line-number" data-line-number="1214"></td>
        <td id="LC1214" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1215" class="blob-num js-line-number" data-line-number="1215"></td>
        <td id="LC1215" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1216" class="blob-num js-line-number" data-line-number="1216"></td>
        <td id="LC1216" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>mouseout</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1217" class="blob-num js-line-number" data-line-number="1217"></td>
        <td id="LC1217" class="blob-code blob-code-inner js-file-line">		<span class=pl-s1>mouseout</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1218" class="blob-num js-line-number" data-line-number="1218"></td>
        <td id="LC1218" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1219" class="blob-num js-line-number" data-line-number="1219"></td>
        <td id="LC1219" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1220" class="blob-num js-line-number" data-line-number="1220"></td>
        <td id="LC1220" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>sliderBar</span><span class=pl-kos>.</span><span class=pl-en>labelFunc</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>_</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1221" class="blob-num js-line-number" data-line-number="1221"></td>
        <td id="LC1221" class="blob-code blob-code-inner js-file-line">		<span class=pl-en>labelFunc</span> <span class=pl-c1>=</span> <span class=pl-s1>_</span><span class=pl-kos>;</span> <span class=pl-k>return</span> <span class=pl-s1>sliderBar</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1222" class="blob-num js-line-number" data-line-number="1222"></td>
        <td id="LC1222" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1223" class="blob-num js-line-number" data-line-number="1223"></td>
        <td id="LC1223" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1224" class="blob-num js-line-number" data-line-number="1224"></td>
        <td id="LC1224" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>return</span> <span class=pl-s1>sliderBar</span></td>
      </tr>
      <tr>
        <td id="L1225" class="blob-num js-line-number" data-line-number="1225"></td>
        <td id="LC1225" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1226" class="blob-num js-line-number" data-line-number="1226"></td>
        <td id="LC1226" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1227" class="blob-num js-line-number" data-line-number="1227"></td>
        <td id="LC1227" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>renderDraggable</span><span class=pl-kos>(</span><span class=pl-s1>svg</span><span class=pl-kos>,</span> <span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>,</span> <span class=pl-s1>radius</span><span class=pl-kos>,</span> <span class=pl-s1>text</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1228" class="blob-num js-line-number" data-line-number="1228"></td>
        <td id="LC1228" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1229" class="blob-num js-line-number" data-line-number="1229"></td>
        <td id="LC1229" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>group</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1230" class="blob-num js-line-number" data-line-number="1230"></td>
        <td id="LC1230" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>path</span> <span class=pl-c1>=</span> <span class=pl-s1>group</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;path&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;none&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1231" class="blob-num js-line-number" data-line-number="1231"></td>
        <td id="LC1231" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>circlePointer</span> <span class=pl-c1>=</span> svg.append(&quot;circle&quot;)</td>
      </tr>
      <tr>
        <td id="L1232" class="blob-num js-line-number" data-line-number="1232"></td>
        <td id="LC1232" class="blob-code blob-code-inner js-file-line">              .<span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1233" class="blob-num js-line-number" data-line-number="1233"></td>
        <td id="LC1233" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>p1</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1234" class="blob-num js-line-number" data-line-number="1234"></td>
        <td id="LC1234" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>radius</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1235" class="blob-num js-line-number" data-line-number="1235"></td>
        <td id="LC1235" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;white&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill-opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1236" class="blob-num js-line-number" data-line-number="1236"></td>
        <td id="LC1236" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1237" class="blob-num js-line-number" data-line-number="1237"></td>
        <td id="LC1237" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1238" class="blob-num js-line-number" data-line-number="1238"></td>
        <td id="LC1238" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1239" class="blob-num js-line-number" data-line-number="1239"></td>
        <td id="LC1239" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>p1</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1240" class="blob-num js-line-number" data-line-number="1240"></td>
        <td id="LC1240" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-en>ringPath</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1241" class="blob-num js-line-number" data-line-number="1241"></td>
        <td id="LC1241" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;d&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>d</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1242" class="blob-num js-line-number" data-line-number="1242"></td>
        <td id="LC1242" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>circlePointer</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1243" class="blob-num js-line-number" data-line-number="1243"></td>
        <td id="LC1243" class="blob-code blob-code-inner js-file-line">                  <span class=pl-smi>console</span><span class=pl-kos>.</span><span class=pl-en>log</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1244" class="blob-num js-line-number" data-line-number="1244"></td>
        <td id="LC1244" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1245" class="blob-num js-line-number" data-line-number="1245"></td>
        <td id="LC1245" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1246" class="blob-num js-line-number" data-line-number="1246"></td>
        <td id="LC1246" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>circleDragger</span> <span class=pl-c1>=</span> svg.append(&quot;circle&quot;)</td>
      </tr>
      <tr>
        <td id="L1247" class="blob-num js-line-number" data-line-number="1247"></td>
        <td id="LC1247" class="blob-code blob-code-inner js-file-line">              .<span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1248" class="blob-num js-line-number" data-line-number="1248"></td>
        <td id="LC1248" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1249" class="blob-num js-line-number" data-line-number="1249"></td>
        <td id="LC1249" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>4</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1250" class="blob-num js-line-number" data-line-number="1250"></td>
        <td id="LC1250" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;white&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;fill-opacity&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;black&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;stroke-width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1251" class="blob-num js-line-number" data-line-number="1251"></td>
        <td id="LC1251" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>drag</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>on</span><span class=pl-kos>(</span><span class=pl-s>&quot;drag&quot;</span><span class=pl-kos>,</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1252" class="blob-num js-line-number" data-line-number="1252"></td>
        <td id="LC1252" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1253" class="blob-num js-line-number" data-line-number="1253"></td>
        <td id="LC1253" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>mouse</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1254" class="blob-num js-line-number" data-line-number="1254"></td>
        <td id="LC1254" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>p2</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1255" class="blob-num js-line-number" data-line-number="1255"></td>
        <td id="LC1255" class="blob-code blob-code-inner js-file-line">                  <span class=pl-k>var</span> <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-en>ringPath</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1256" class="blob-num js-line-number" data-line-number="1256"></td>
        <td id="LC1256" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;d&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>d</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1257" class="blob-num js-line-number" data-line-number="1257"></td>
        <td id="LC1257" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>label</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>label</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>label</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span>  <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1258" class="blob-num js-line-number" data-line-number="1258"></td>
        <td id="LC1258" class="blob-code blob-code-inner js-file-line">                  <span class=pl-s1>circleDragger</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span><span class=pl-s1>y</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1259" class="blob-num js-line-number" data-line-number="1259"></td>
        <td id="LC1259" class="blob-code blob-code-inner js-file-line">                  <span class=pl-smi>console</span><span class=pl-kos>.</span><span class=pl-en>log</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1260" class="blob-num js-line-number" data-line-number="1260"></td>
        <td id="LC1260" class="blob-code blob-code-inner js-file-line">              <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1261" class="blob-num js-line-number" data-line-number="1261"></td>
        <td id="LC1261" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1262" class="blob-num js-line-number" data-line-number="1262"></td>
        <td id="LC1262" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>label</span> <span class=pl-c1>=</span> svg.append(&quot;text&quot;)</td>
      </tr>
      <tr>
        <td id="L1263" class="blob-num js-line-number" data-line-number="1263"></td>
        <td id="LC1263" class="blob-code blob-code-inner js-file-line">      .style(&quot;position&quot;, &quot;absolute&quot;)</td>
      </tr>
      <tr>
        <td id="L1264" class="blob-num js-line-number" data-line-number="1264"></td>
        <td id="LC1264" class="blob-code blob-code-inner js-file-line">      .style(&quot;border-radius&quot;, &quot;3px&quot;)</td>
      </tr>
      <tr>
        <td id="L1265" class="blob-num js-line-number" data-line-number="1265"></td>
        <td id="LC1265" class="blob-code blob-code-inner js-file-line">      .style(&quot;text-align&quot;, &quot;start&quot;)</td>
      </tr>
      <tr>
        <td id="L1266" class="blob-num js-line-number" data-line-number="1266"></td>
        <td id="LC1266" class="blob-code blob-code-inner js-file-line">      .<span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;padding-top&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;5px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1267" class="blob-num js-line-number" data-line-number="1267"></td>
        <td id="LC1267" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;figtext&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1268" class="blob-num js-line-number" data-line-number="1268"></td>
        <td id="LC1268" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1269" class="blob-num js-line-number" data-line-number="1269"></td>
        <td id="LC1269" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1270" class="blob-num js-line-number" data-line-number="1270"></td>
        <td id="LC1270" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>100</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1271" class="blob-num js-line-number" data-line-number="1271"></td>
        <td id="LC1271" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>10</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1272" class="blob-num js-line-number" data-line-number="1272"></td>
        <td id="LC1272" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;r&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>7</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1273" class="blob-num js-line-number" data-line-number="1273"></td>
        <td id="LC1273" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>html</span><span class=pl-kos>(</span><span class=pl-s1>text</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1274" class="blob-num js-line-number" data-line-number="1274"></td>
        <td id="LC1274" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1275" class="blob-num js-line-number" data-line-number="1275"></td>
        <td id="LC1275" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>ringPath</span> <span class=pl-c1>=</span> <span class=pl-en>ringPathGen</span><span class=pl-kos>(</span><span class=pl-s1>radius</span><span class=pl-kos>,</span> <span class=pl-s1>label</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>getBBox</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-c1>width</span><span class=pl-kos>,</span> <span class=pl-s1>label</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>getBBox</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-c1>height</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1276" class="blob-num js-line-number" data-line-number="1276"></td>
        <td id="LC1276" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1277" class="blob-num js-line-number" data-line-number="1277"></td>
        <td id="LC1277" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s1>ringPath</span><span class=pl-kos>(</span><span class=pl-s1>p1</span><span class=pl-kos>,</span> <span class=pl-s1>p2</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1278" class="blob-num js-line-number" data-line-number="1278"></td>
        <td id="LC1278" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>path</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;d&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>d</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1279" class="blob-num js-line-number" data-line-number="1279"></td>
        <td id="LC1279" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>label</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;translate(&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>label</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;,&quot;</span> <span class=pl-c1>+</span> <span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>label</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span>  <span class=pl-c1>+</span> <span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1280" class="blob-num js-line-number" data-line-number="1280"></td>
        <td id="LC1280" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>circleDragger</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cx&quot;</span><span class=pl-kos>,</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;cy&quot;</span><span class=pl-kos>,</span><span class=pl-s1>p2</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1281" class="blob-num js-line-number" data-line-number="1281"></td>
        <td id="LC1281" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1282" class="blob-num js-line-number" data-line-number="1282"></td>
        <td id="LC1282" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>group</span></td>
      </tr>
      <tr>
        <td id="L1283" class="blob-num js-line-number" data-line-number="1283"></td>
        <td id="LC1283" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1284" class="blob-num js-line-number" data-line-number="1284"></td>
        <td id="LC1284" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1285" class="blob-num js-line-number" data-line-number="1285"></td>
        <td id="LC1285" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/****************************************************************************</span></td>
      </tr>
      <tr>
        <td id="L1286" class="blob-num js-line-number" data-line-number="1286"></td>
        <td id="LC1286" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  MISC MATH AND JAVASCRIPT HELPERS</span></td>
      </tr>
      <tr>
        <td id="L1287" class="blob-num js-line-number" data-line-number="1287"></td>
        <td id="LC1287" class="blob-code blob-code-inner js-file-line"><span class=pl-c>****************************************************************************/</span></td>
      </tr>
      <tr>
        <td id="L1288" class="blob-num js-line-number" data-line-number="1288"></td>
        <td id="LC1288" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1289" class="blob-num js-line-number" data-line-number="1289"></td>
        <td id="LC1289" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Parse a color string from rgb(0,0,0) format */</span></td>
      </tr>
      <tr>
        <td id="L1290" class="blob-num js-line-number" data-line-number="1290"></td>
        <td id="LC1290" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>parseColor</span><span class=pl-kos>(</span><span class=pl-s1>input</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1291" class="blob-num js-line-number" data-line-number="1291"></td>
        <td id="LC1291" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>input</span><span class=pl-kos>.</span><span class=pl-en>split</span><span class=pl-kos>(</span><span class=pl-s>&quot;(&quot;</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>split</span><span class=pl-kos>(</span><span class=pl-s>&quot;)&quot;</span><span class=pl-kos>)</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>split</span><span class=pl-kos>(</span><span class=pl-s>&quot;,&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>map</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>)</span><span class=pl-kos>{</span><span class=pl-k>return</span> <span class=pl-en>parseInt</span><span class=pl-kos>(</span><span class=pl-s1>i</span><span class=pl-kos>.</span><span class=pl-en>trim</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1292" class="blob-num js-line-number" data-line-number="1292"></td>
        <td id="LC1292" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1293" class="blob-num js-line-number" data-line-number="1293"></td>
        <td id="LC1293" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1294" class="blob-num js-line-number" data-line-number="1294"></td>
        <td id="LC1294" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Rosenbrok Function banana function */</span></td>
      </tr>
      <tr>
        <td id="L1295" class="blob-num js-line-number" data-line-number="1295"></td>
        <td id="LC1295" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>bananaf</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1296" class="blob-num js-line-number" data-line-number="1296"></td>
        <td id="LC1296" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>s</span> <span class=pl-c1>=</span> <span class=pl-c1>3</span></td>
      </tr>
      <tr>
        <td id="L1297" class="blob-num js-line-number" data-line-number="1297"></td>
        <td id="LC1297" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span>*<span class=pl-s1>s</span></td>
      </tr>
      <tr>
        <td id="L1298" class="blob-num js-line-number" data-line-number="1298"></td>
        <td id="LC1298" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fx</span>   <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-c1>20</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-kos>)</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1299" class="blob-num js-line-number" data-line-number="1299"></td>
        <td id="LC1299" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>dfx</span>  <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>2</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-c1>80</span>*<span class=pl-s1>x</span>*<span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-s1>s</span>*<span class=pl-c1>40</span>*<span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1300" class="blob-num js-line-number" data-line-number="1300"></td>
        <td id="LC1300" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>d2fx</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span> <span class=pl-kos>[</span> <span class=pl-c1>2</span> <span class=pl-c1>+</span> <span class=pl-c1>160</span>*<span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-c1>-</span> <span class=pl-c1>80</span>*<span class=pl-kos>(</span><span class=pl-c1>-</span><span class=pl-s1>x</span>*<span class=pl-s1>x</span> <span class=pl-c1>+</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-c1>-</span><span class=pl-c1>80</span>*<span class=pl-s1>x</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-kos>[</span> <span class=pl-c1>-</span><span class=pl-c1>80</span>*<span class=pl-s1>x</span><span class=pl-kos>,</span> <span class=pl-c1>40</span>  <span class=pl-kos>]</span> <span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1301" class="blob-num js-line-number" data-line-number="1301"></td>
        <td id="LC1301" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-s1>fx</span>*<span class=pl-c1>1</span><span class=pl-kos>,</span> <span class=pl-s1>dfx</span><span class=pl-kos>,</span> <span class=pl-s1>d2fx</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1302" class="blob-num js-line-number" data-line-number="1302"></td>
        <td id="LC1302" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1303" class="blob-num js-line-number" data-line-number="1303"></td>
        <td id="LC1303" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1304" class="blob-num js-line-number" data-line-number="1304"></td>
        <td id="LC1304" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Nonsmooth variation on Rosenbrok Banana Function */</span></td>
      </tr>
      <tr>
        <td id="L1305" class="blob-num js-line-number" data-line-number="1305"></td>
        <td id="LC1305" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>bananaabsf</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1306" class="blob-num js-line-number" data-line-number="1306"></td>
        <td id="LC1306" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>x</span> <span class=pl-c1>=</span> <span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-k>var</span> <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>xy</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1307" class="blob-num js-line-number" data-line-number="1307"></td>
        <td id="LC1307" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fx</span>   <span class=pl-c1>=</span> <span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>+</span> <span class=pl-c1>20</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-kos>)</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1308" class="blob-num js-line-number" data-line-number="1308"></td>
        <td id="LC1308" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>dfx</span>  <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-c1>-</span><span class=pl-c1>2</span>*<span class=pl-kos>(</span><span class=pl-c1>1</span><span class=pl-c1>-</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-c1>-</span> <span class=pl-c1>20</span>*<span class=pl-kos>(</span><span class=pl-kos>(</span><span class=pl-s1>x</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span><span class=pl-kos>)</span> ? <span class=pl-c1>1</span> : <span class=pl-c1>-</span><span class=pl-c1>1</span><span class=pl-kos>)</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>,</span> <span class=pl-c1>20</span>*<span class=pl-kos>(</span><span class=pl-s1>y</span> <span class=pl-c1>-</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>abs</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1309" class="blob-num js-line-number" data-line-number="1309"></td>
        <td id="LC1309" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-s1>fx</span><span class=pl-kos>,</span> <span class=pl-s1>dfx</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1310" class="blob-num js-line-number" data-line-number="1310"></td>
        <td id="LC1310" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1311" class="blob-num js-line-number" data-line-number="1311"></td>
        <td id="LC1311" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1312" class="blob-num js-line-number" data-line-number="1312"></td>
        <td id="LC1312" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Rotated Quadratic */</span></td>
      </tr>
      <tr>
        <td id="L1313" class="blob-num js-line-number" data-line-number="1313"></td>
        <td id="LC1313" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>quadf</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1314" class="blob-num js-line-number" data-line-number="1314"></td>
        <td id="LC1314" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>U</span> <span class=pl-c1>=</span> <span class=pl-en>givens</span><span class=pl-kos>(</span><span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-c1>PI</span>/<span class=pl-c1>4</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1315" class="blob-num js-line-number" data-line-number="1315"></td>
        <td id="LC1315" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>lambda</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-c1>100</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1316" class="blob-num js-line-number" data-line-number="1316"></td>
        <td id="LC1316" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>A</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>,</span> <span class=pl-v>U</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1317" class="blob-num js-line-number" data-line-number="1317"></td>
        <td id="LC1317" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>dfx</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>A</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1318" class="blob-num js-line-number" data-line-number="1318"></td>
        <td id="LC1318" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fx</span>  <span class=pl-c1>=</span> <span class=pl-c1>0.5</span>*<span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>dfx</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1319" class="blob-num js-line-number" data-line-number="1319"></td>
        <td id="LC1319" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-s1>fx</span><span class=pl-kos>,</span> <span class=pl-s1>dfx</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1320" class="blob-num js-line-number" data-line-number="1320"></td>
        <td id="LC1320" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1321" class="blob-num js-line-number" data-line-number="1321"></td>
        <td id="LC1321" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1322" class="blob-num js-line-number" data-line-number="1322"></td>
        <td id="LC1322" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Identity */</span></td>
      </tr>
      <tr>
        <td id="L1323" class="blob-num js-line-number" data-line-number="1323"></td>
        <td id="LC1323" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>eyef</span><span class=pl-kos>(</span><span class=pl-s1>xy</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1324" class="blob-num js-line-number" data-line-number="1324"></td>
        <td id="LC1324" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>U</span> <span class=pl-c1>=</span> <span class=pl-en>givens</span><span class=pl-kos>(</span><span class=pl-c1>0</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1325" class="blob-num js-line-number" data-line-number="1325"></td>
        <td id="LC1325" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>lambda</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>diag</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>,</span><span class=pl-c1>100</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1326" class="blob-num js-line-number" data-line-number="1326"></td>
        <td id="LC1326" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>A</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>)</span><span class=pl-kos>,</span><span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>,</span> <span class=pl-v>U</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1327" class="blob-num js-line-number" data-line-number="1327"></td>
        <td id="LC1327" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>dfx</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-v>A</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1328" class="blob-num js-line-number" data-line-number="1328"></td>
        <td id="LC1328" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>fx</span>  <span class=pl-c1>=</span> <span class=pl-c1>0.5</span>*<span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>dot</span><span class=pl-kos>(</span><span class=pl-s1>dfx</span><span class=pl-kos>,</span><span class=pl-s1>xy</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1329" class="blob-num js-line-number" data-line-number="1329"></td>
        <td id="LC1329" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-s1>fx</span><span class=pl-kos>,</span> <span class=pl-s1>dfx</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1330" class="blob-num js-line-number" data-line-number="1330"></td>
        <td id="LC1330" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1331" class="blob-num js-line-number" data-line-number="1331"></td>
        <td id="LC1331" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1332" class="blob-num js-line-number" data-line-number="1332"></td>
        <td id="LC1332" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Givens rotations */</span></td>
      </tr>
      <tr>
        <td id="L1333" class="blob-num js-line-number" data-line-number="1333"></td>
        <td id="LC1333" class="blob-code blob-code-inner js-file-line"><span class=pl-k>var</span> <span class=pl-en>givens</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>theta</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1334" class="blob-num js-line-number" data-line-number="1334"></td>
        <td id="LC1334" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>c</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>cos</span><span class=pl-kos>(</span><span class=pl-s1>theta</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1335" class="blob-num js-line-number" data-line-number="1335"></td>
        <td id="LC1335" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>s</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>sin</span><span class=pl-kos>(</span><span class=pl-s1>theta</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1336" class="blob-num js-line-number" data-line-number="1336"></td>
        <td id="LC1336" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>[</span><span class=pl-kos>[</span><span class=pl-s1>c</span><span class=pl-kos>,</span> <span class=pl-c1>-</span><span class=pl-s1>s</span><span class=pl-kos>]</span><span class=pl-kos>,</span> <span class=pl-kos>[</span><span class=pl-s1>s</span><span class=pl-kos>,</span> <span class=pl-s1>c</span><span class=pl-kos>]</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1337" class="blob-num js-line-number" data-line-number="1337"></td>
        <td id="LC1337" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1338" class="blob-num js-line-number" data-line-number="1338"></td>
        <td id="LC1338" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1339" class="blob-num js-line-number" data-line-number="1339"></td>
        <td id="LC1339" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Global controller for float -&gt; string conversion */</span></td>
      </tr>
      <tr>
        <td id="L1340" class="blob-num js-line-number" data-line-number="1340"></td>
        <td id="LC1340" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>round</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1341" class="blob-num js-line-number" data-line-number="1341"></td>
        <td id="LC1341" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>toPrecision</span><span class=pl-kos>(</span><span class=pl-c1>3</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1342" class="blob-num js-line-number" data-line-number="1342"></td>
        <td id="LC1342" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1343" class="blob-num js-line-number" data-line-number="1343"></td>
        <td id="LC1343" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1344" class="blob-num js-line-number" data-line-number="1344"></td>
        <td id="LC1344" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/* Moves a svg element to the front */</span></td>
      </tr>
      <tr>
        <td id="L1345" class="blob-num js-line-number" data-line-number="1345"></td>
        <td id="LC1345" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-c1>selection</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>moveToFront</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1346" class="blob-num js-line-number" data-line-number="1346"></td>
        <td id="LC1346" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>each</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1347" class="blob-num js-line-number" data-line-number="1347"></td>
        <td id="LC1347" class="blob-code blob-code-inner js-file-line">    <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-en>appendChild</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1348" class="blob-num js-line-number" data-line-number="1348"></td>
        <td id="LC1348" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1349" class="blob-num js-line-number" data-line-number="1349"></td>
        <td id="LC1349" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1350" class="blob-num js-line-number" data-line-number="1350"></td>
        <td id="LC1350" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1351" class="blob-num js-line-number" data-line-number="1351"></td>
        <td id="LC1351" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-c1>selection</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-en>moveToBack</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1352" class="blob-num js-line-number" data-line-number="1352"></td>
        <td id="LC1352" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-en>each</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1353" class="blob-num js-line-number" data-line-number="1353"></td>
        <td id="LC1353" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>var</span> <span class=pl-s1>firstChild</span> <span class=pl-c1>=</span> <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-c1>firstChild</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1354" class="blob-num js-line-number" data-line-number="1354"></td>
        <td id="LC1354" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>firstChild</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1355" class="blob-num js-line-number" data-line-number="1355"></td>
        <td id="LC1355" class="blob-code blob-code-inner js-file-line">            <span class=pl-smi>this</span><span class=pl-kos>.</span><span class=pl-c1>parentNode</span><span class=pl-kos>.</span><span class=pl-en>insertBefore</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>,</span> <span class=pl-s1>firstChild</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1356" class="blob-num js-line-number" data-line-number="1356"></td>
        <td id="LC1356" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1357" class="blob-num js-line-number" data-line-number="1357"></td>
        <td id="LC1357" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1358" class="blob-num js-line-number" data-line-number="1358"></td>
        <td id="LC1358" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1359" class="blob-num js-line-number" data-line-number="1359"></td>
        <td id="LC1359" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1360" class="blob-num js-line-number" data-line-number="1360"></td>
        <td id="LC1360" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1361" class="blob-num js-line-number" data-line-number="1361"></td>
        <td id="LC1361" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Generates array of zeros</span></td>
      </tr>
      <tr>
        <td id="L1362" class="blob-num js-line-number" data-line-number="1362"></td>
        <td id="LC1362" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1363" class="blob-num js-line-number" data-line-number="1363"></td>
        <td id="LC1363" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>ones</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1364" class="blob-num js-line-number" data-line-number="1364"></td>
        <td id="LC1364" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-v>Array</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-v>Array</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>map</span><span class=pl-kos>(</span><span class=pl-v>Number</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-c1>valueOf</span><span class=pl-kos>,</span><span class=pl-c1>1</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1365" class="blob-num js-line-number" data-line-number="1365"></td>
        <td id="LC1365" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1366" class="blob-num js-line-number" data-line-number="1366"></td>
        <td id="LC1366" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1367" class="blob-num js-line-number" data-line-number="1367"></td>
        <td id="LC1367" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1368" class="blob-num js-line-number" data-line-number="1368"></td>
        <td id="LC1368" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Generates array of zeros</span></td>
      </tr>
      <tr>
        <td id="L1369" class="blob-num js-line-number" data-line-number="1369"></td>
        <td id="LC1369" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1370" class="blob-num js-line-number" data-line-number="1370"></td>
        <td id="LC1370" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>zeros</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1371" class="blob-num js-line-number" data-line-number="1371"></td>
        <td id="LC1371" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-v>Array</span><span class=pl-kos>.</span><span class=pl-en>apply</span><span class=pl-kos>(</span>null<span class=pl-kos>,</span> <span class=pl-v>Array</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>map</span><span class=pl-kos>(</span><span class=pl-v>Number</span><span class=pl-kos>.</span><span class=pl-c1>prototype</span><span class=pl-kos>.</span><span class=pl-c1>valueOf</span><span class=pl-kos>,</span><span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1372" class="blob-num js-line-number" data-line-number="1372"></td>
        <td id="LC1372" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1373" class="blob-num js-line-number" data-line-number="1373"></td>
        <td id="LC1373" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1374" class="blob-num js-line-number" data-line-number="1374"></td>
        <td id="LC1374" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1375" class="blob-num js-line-number" data-line-number="1375"></td>
        <td id="LC1375" class="blob-code blob-code-inner js-file-line"><span class=pl-c>  Generates array of zeros</span></td>
      </tr>
      <tr>
        <td id="L1376" class="blob-num js-line-number" data-line-number="1376"></td>
        <td id="LC1376" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1377" class="blob-num js-line-number" data-line-number="1377"></td>
        <td id="LC1377" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>zeros2D</span><span class=pl-kos>(</span><span class=pl-s1>n</span><span class=pl-kos>,</span><span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1378" class="blob-num js-line-number" data-line-number="1378"></td>
        <td id="LC1378" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>A</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1379" class="blob-num js-line-number" data-line-number="1379"></td>
        <td id="LC1379" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>n</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1380" class="blob-num js-line-number" data-line-number="1380"></td>
        <td id="LC1380" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>A</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-en>zeros</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1381" class="blob-num js-line-number" data-line-number="1381"></td>
        <td id="LC1381" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1382" class="blob-num js-line-number" data-line-number="1382"></td>
        <td id="LC1382" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-v>A</span></td>
      </tr>
      <tr>
        <td id="L1383" class="blob-num js-line-number" data-line-number="1383"></td>
        <td id="LC1383" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1384" class="blob-num js-line-number" data-line-number="1384"></td>
        <td id="LC1384" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1385" class="blob-num js-line-number" data-line-number="1385"></td>
        <td id="LC1385" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1386" class="blob-num js-line-number" data-line-number="1386"></td>
        <td id="LC1386" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1387" class="blob-num js-line-number" data-line-number="1387"></td>
        <td id="LC1387" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Create Vandermonde matrix of size x and order degree</span></td>
      </tr>
      <tr>
        <td id="L1388" class="blob-num js-line-number" data-line-number="1388"></td>
        <td id="LC1388" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1389" class="blob-num js-line-number" data-line-number="1389"></td>
        <td id="LC1389" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>vandermonde</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>,</span> <span class=pl-s1>degree</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1390" class="blob-num js-line-number" data-line-number="1390"></td>
        <td id="LC1390" class="blob-code blob-code-inner js-file-line">	<span class=pl-v>A</span> <span class=pl-c1>=</span> <span class=pl-en>zeros2D</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>,</span><span class=pl-s1>degree</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1391" class="blob-num js-line-number" data-line-number="1391"></td>
        <td id="LC1391" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>++</span><span class=pl-kos>)</span><span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1392" class="blob-num js-line-number" data-line-number="1392"></td>
        <td id="LC1392" class="blob-code blob-code-inner js-file-line">	  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>j</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>degree</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span><span class=pl-kos>;</span> <span class=pl-s1>j</span> <span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1393" class="blob-num js-line-number" data-line-number="1393"></td>
        <td id="LC1393" class="blob-code blob-code-inner js-file-line">	    <span class=pl-v>A</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-s1>j</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>,</span><span class=pl-s1>j</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1394" class="blob-num js-line-number" data-line-number="1394"></td>
        <td id="LC1394" class="blob-code blob-code-inner js-file-line">	  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1395" class="blob-num js-line-number" data-line-number="1395"></td>
        <td id="LC1395" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1396" class="blob-num js-line-number" data-line-number="1396"></td>
        <td id="LC1396" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>return</span> <span class=pl-v>A</span></td>
      </tr>
      <tr>
        <td id="L1397" class="blob-num js-line-number" data-line-number="1397"></td>
        <td id="LC1397" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1398" class="blob-num js-line-number" data-line-number="1398"></td>
        <td id="LC1398" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1399" class="blob-num js-line-number" data-line-number="1399"></td>
        <td id="LC1399" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1400" class="blob-num js-line-number" data-line-number="1400"></td>
        <td id="LC1400" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Evaluate a 1D polynomial</span></td>
      </tr>
      <tr>
        <td id="L1401" class="blob-num js-line-number" data-line-number="1401"></td>
        <td id="LC1401" class="blob-code blob-code-inner js-file-line"><span class=pl-c>w[0]x[0] + ... + w[k]x[k], k = w.length</span></td>
      </tr>
      <tr>
        <td id="L1402" class="blob-num js-line-number" data-line-number="1402"></td>
        <td id="LC1402" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1403" class="blob-num js-line-number" data-line-number="1403"></td>
        <td id="LC1403" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>poly</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>,</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1404" class="blob-num js-line-number" data-line-number="1404"></td>
        <td id="LC1404" class="blob-code blob-code-inner js-file-line">	<span class=pl-s1>s</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L1405" class="blob-num js-line-number" data-line-number="1405"></td>
        <td id="LC1405" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-s1>w</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-s1>s</span> <span class=pl-c1>=</span> <span class=pl-s1>s</span> <span class=pl-c1>+</span> <span class=pl-s1>w</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span>*<span class=pl-v>Math</span><span class=pl-kos>.</span><span class=pl-en>pow</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>,</span><span class=pl-s1>i</span><span class=pl-kos>)</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1406" class="blob-num js-line-number" data-line-number="1406"></td>
        <td id="LC1406" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>return</span> <span class=pl-s1>s</span></td>
      </tr>
      <tr>
        <td id="L1407" class="blob-num js-line-number" data-line-number="1407"></td>
        <td id="LC1407" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1408" class="blob-num js-line-number" data-line-number="1408"></td>
        <td id="LC1408" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1409" class="blob-num js-line-number" data-line-number="1409"></td>
        <td id="LC1409" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1410" class="blob-num js-line-number" data-line-number="1410"></td>
        <td id="LC1410" class="blob-code blob-code-inner js-file-line"><span class=pl-c>/*</span></td>
      </tr>
      <tr>
        <td id="L1411" class="blob-num js-line-number" data-line-number="1411"></td>
        <td id="LC1411" class="blob-code blob-code-inner js-file-line"><span class=pl-c>Evaluates the polynomial in range [-1.1, 1.1] at 1800 intervals</span></td>
      </tr>
      <tr>
        <td id="L1412" class="blob-num js-line-number" data-line-number="1412"></td>
        <td id="LC1412" class="blob-code blob-code-inner js-file-line"><span class=pl-c>*/</span></td>
      </tr>
      <tr>
        <td id="L1413" class="blob-num js-line-number" data-line-number="1413"></td>
        <td id="LC1413" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>evalPoly</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1414" class="blob-num js-line-number" data-line-number="1414"></td>
        <td id="LC1414" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>var</span> <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1415" class="blob-num js-line-number" data-line-number="1415"></td>
        <td id="LC1415" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-c1>900</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>900</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1416" class="blob-num js-line-number" data-line-number="1416"></td>
        <td id="LC1416" class="blob-code blob-code-inner js-file-line">	  <span class=pl-s1>data</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-kos>[</span><span class=pl-s1>i</span>/<span class=pl-c1>800</span><span class=pl-kos>,</span> <span class=pl-c1>1</span>*<span class=pl-en>poly</span><span class=pl-kos>(</span><span class=pl-s1>w</span><span class=pl-kos>,</span> <span class=pl-s1>i</span>/<span class=pl-c1>800</span><span class=pl-kos>)</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1417" class="blob-num js-line-number" data-line-number="1417"></td>
        <td id="LC1417" class="blob-code blob-code-inner js-file-line">	<span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1418" class="blob-num js-line-number" data-line-number="1418"></td>
        <td id="LC1418" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>return</span> <span class=pl-s1>data</span></td>
      </tr>
      <tr>
        <td id="L1419" class="blob-num js-line-number" data-line-number="1419"></td>
        <td id="LC1419" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1420" class="blob-num js-line-number" data-line-number="1420"></td>
        <td id="LC1420" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1421" class="blob-num js-line-number" data-line-number="1421"></td>
        <td id="LC1421" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1422" class="blob-num js-line-number" data-line-number="1422"></td>
        <td id="LC1422" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>setTM</span><span class=pl-kos>(</span><span class=pl-s1>element</span><span class=pl-kos>,</span> <span class=pl-s1>m</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1423" class="blob-num js-line-number" data-line-number="1423"></td>
        <td id="LC1423" class="blob-code blob-code-inner js-file-line">	<span class=pl-k>return</span> <span class=pl-s1>element</span><span class=pl-kos>.</span><span class=pl-c1>transform</span><span class=pl-kos>.</span><span class=pl-c1>baseVal</span><span class=pl-kos>.</span><span class=pl-en>initialize</span><span class=pl-kos>(</span><span class=pl-s1>element</span><span class=pl-kos>.</span><span class=pl-c1>ownerSVGElement</span><span class=pl-kos>.</span><span class=pl-en>createSVGTransformFromMatrix</span><span class=pl-kos>(</span><span class=pl-s1>m</span><span class=pl-kos>)</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1424" class="blob-num js-line-number" data-line-number="1424"></td>
        <td id="LC1424" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1425" class="blob-num js-line-number" data-line-number="1425"></td>
        <td id="LC1425" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1426" class="blob-num js-line-number" data-line-number="1426"></td>
        <td id="LC1426" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1427" class="blob-num js-line-number" data-line-number="1427"></td>
        <td id="LC1427" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>wrap</span><span class=pl-kos>(</span><span class=pl-s1>text</span><span class=pl-kos>,</span> <span class=pl-s1>width</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1428" class="blob-num js-line-number" data-line-number="1428"></td>
        <td id="LC1428" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>each</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1429" class="blob-num js-line-number" data-line-number="1429"></td>
        <td id="LC1429" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>text</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>select</span><span class=pl-kos>(</span><span class=pl-smi>this</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1430" class="blob-num js-line-number" data-line-number="1430"></td>
        <td id="LC1430" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>words</span> <span class=pl-c1>=</span> <span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>split</span><span class=pl-kos>(</span><span class=pl-pds>/<span class=pl-cce>\s</span><span class=pl-c1>+</span>/</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>reverse</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1431" class="blob-num js-line-number" data-line-number="1431"></td>
        <td id="LC1431" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>word</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1432" class="blob-num js-line-number" data-line-number="1432"></td>
        <td id="LC1432" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>line</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1433" class="blob-num js-line-number" data-line-number="1433"></td>
        <td id="LC1433" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>lineNumber</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1434" class="blob-num js-line-number" data-line-number="1434"></td>
        <td id="LC1434" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>lineHeight</span> <span class=pl-c1>=</span> <span class=pl-c1>1.1</span><span class=pl-kos>,</span> <span class=pl-c>// ems</span></td>
      </tr>
      <tr>
        <td id="L1435" class="blob-num js-line-number" data-line-number="1435"></td>
        <td id="LC1435" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1436" class="blob-num js-line-number" data-line-number="1436"></td>
        <td id="LC1436" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dy</span> <span class=pl-c1>=</span> <span class=pl-en>parseFloat</span><span class=pl-kos>(</span><span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;dy&quot;</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>,</span></td>
      </tr>
      <tr>
        <td id="L1437" class="blob-num js-line-number" data-line-number="1437"></td>
        <td id="LC1437" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>tspan</span> <span class=pl-c1>=</span> <span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span>null<span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;tspan&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;dy&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>dy</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;em&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1438" class="blob-num js-line-number" data-line-number="1438"></td>
        <td id="LC1438" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>while</span> <span class=pl-kos>(</span><span class=pl-s1>word</span> <span class=pl-c1>=</span> <span class=pl-s1>words</span><span class=pl-kos>.</span><span class=pl-en>pop</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1439" class="blob-num js-line-number" data-line-number="1439"></td>
        <td id="LC1439" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-s1>word</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1440" class="blob-num js-line-number" data-line-number="1440"></td>
        <td id="LC1440" class="blob-code blob-code-inner js-file-line">      <span class=pl-s1>tspan</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span><span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>join</span><span class=pl-kos>(</span><span class=pl-s>&quot; &quot;</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1441" class="blob-num js-line-number" data-line-number="1441"></td>
        <td id="LC1441" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-kos>(</span><span class=pl-s1>tspan</span><span class=pl-kos>.</span><span class=pl-en>node</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>getComputedTextLength</span><span class=pl-kos>(</span><span class=pl-kos>)</span> <span class=pl-c1>&gt;</span> <span class=pl-s1>width</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1442" class="blob-num js-line-number" data-line-number="1442"></td>
        <td id="LC1442" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>pop</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1443" class="blob-num js-line-number" data-line-number="1443"></td>
        <td id="LC1443" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>tspan</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span><span class=pl-s1>line</span><span class=pl-kos>.</span><span class=pl-en>join</span><span class=pl-kos>(</span><span class=pl-s>&quot; &quot;</span><span class=pl-kos>)</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1444" class="blob-num js-line-number" data-line-number="1444"></td>
        <td id="LC1444" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>line</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-s1>word</span><span class=pl-kos>]</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1445" class="blob-num js-line-number" data-line-number="1445"></td>
        <td id="LC1445" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>tspan</span> <span class=pl-c1>=</span> <span class=pl-s1>text</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;tspan&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;x&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>0</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;y&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>y</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;dy&quot;</span><span class=pl-kos>,</span> <span class=pl-c1>++</span><span class=pl-s1>lineNumber</span> * <span class=pl-s1>lineHeight</span> <span class=pl-c1>+</span> <span class=pl-s1>dy</span> <span class=pl-c1>+</span> <span class=pl-s>&quot;em&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>text</span><span class=pl-kos>(</span><span class=pl-s1>word</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1446" class="blob-num js-line-number" data-line-number="1446"></td>
        <td id="LC1446" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1447" class="blob-num js-line-number" data-line-number="1447"></td>
        <td id="LC1447" class="blob-code blob-code-inner js-file-line">    <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1448" class="blob-num js-line-number" data-line-number="1448"></td>
        <td id="LC1448" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1449" class="blob-num js-line-number" data-line-number="1449"></td>
        <td id="LC1449" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1450" class="blob-num js-line-number" data-line-number="1450"></td>
        <td id="LC1450" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1451" class="blob-num js-line-number" data-line-number="1451"></td>
        <td id="LC1451" class="blob-code blob-code-inner js-file-line"><span class=pl-k>var</span> <span class=pl-en>inv</span> <span class=pl-c1>=</span> <span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>lambda</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-c1>1</span>/<span class=pl-s1>lambda</span> <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1452" class="blob-num js-line-number" data-line-number="1452"></td>
        <td id="LC1452" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1453" class="blob-num js-line-number" data-line-number="1453"></td>
        <td id="LC1453" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>eigSym</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1454" class="blob-num js-line-number" data-line-number="1454"></td>
        <td id="LC1454" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Eig</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>eig</span><span class=pl-kos>(</span><span class=pl-v>X</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1455" class="blob-num js-line-number" data-line-number="1455"></td>
        <td id="LC1455" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-s1>lambda</span> <span class=pl-c1>=</span> <span class=pl-v>Eig</span><span class=pl-kos>.</span><span class=pl-c1>lambda</span><span class=pl-kos>.</span><span class=pl-c1>x</span></td>
      </tr>
      <tr>
        <td id="L1456" class="blob-num js-line-number" data-line-number="1456"></td>
        <td id="LC1456" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>U</span> <span class=pl-c1>=</span> <span class=pl-s1>numeric</span><span class=pl-kos>.</span><span class=pl-en>transpose</span><span class=pl-kos>(</span><span class=pl-v>Eig</span><span class=pl-kos>.</span><span class=pl-c1>E</span><span class=pl-kos>.</span><span class=pl-c1>x</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1457" class="blob-num js-line-number" data-line-number="1457"></td>
        <td id="LC1457" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>var</span> <span class=pl-v>Z</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>zip</span><span class=pl-kos>(</span><span class=pl-v>U</span><span class=pl-kos>,</span> <span class=pl-s1>lambda</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1458" class="blob-num js-line-number" data-line-number="1458"></td>
        <td id="LC1458" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>Z</span><span class=pl-kos>.</span><span class=pl-en>sort</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>a</span><span class=pl-kos>,</span> <span class=pl-s1>b</span><span class=pl-kos>)</span> <span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-s1>b</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span> <span class=pl-c1>-</span> <span class=pl-s1>a</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>;</span> <span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1459" class="blob-num js-line-number" data-line-number="1459"></td>
        <td id="LC1459" class="blob-code blob-code-inner js-file-line">  <span class=pl-v>U</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1460" class="blob-num js-line-number" data-line-number="1460"></td>
        <td id="LC1460" class="blob-code blob-code-inner js-file-line">  <span class=pl-s1>lambda</span> <span class=pl-c1>=</span> <span class=pl-kos>[</span><span class=pl-kos>]</span></td>
      </tr>
      <tr>
        <td id="L1461" class="blob-num js-line-number" data-line-number="1461"></td>
        <td id="LC1461" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>for</span> <span class=pl-kos>(</span><span class=pl-k>var</span> <span class=pl-s1>i</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span><span class=pl-kos>;</span> <span class=pl-s1>i</span> <span class=pl-c1>&lt;</span> <span class=pl-v>Z</span><span class=pl-kos>.</span><span class=pl-c1>length</span><span class=pl-kos>;</span> <span class=pl-s1>i</span><span class=pl-c1>++</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1462" class="blob-num js-line-number" data-line-number="1462"></td>
        <td id="LC1462" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>U</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-v>Z</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1463" class="blob-num js-line-number" data-line-number="1463"></td>
        <td id="LC1463" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>lambda</span><span class=pl-kos>.</span><span class=pl-en>push</span><span class=pl-kos>(</span><span class=pl-v>Z</span><span class=pl-kos>[</span><span class=pl-s1>i</span><span class=pl-kos>]</span><span class=pl-kos>[</span><span class=pl-c1>1</span><span class=pl-kos>]</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1464" class="blob-num js-line-number" data-line-number="1464"></td>
        <td id="LC1464" class="blob-code blob-code-inner js-file-line">  <span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1465" class="blob-num js-line-number" data-line-number="1465"></td>
        <td id="LC1465" class="blob-code blob-code-inner js-file-line">  <span class=pl-k>return</span> <span class=pl-kos>{</span><span class=pl-c1>U</span>:<span class=pl-v>U</span><span class=pl-kos>,</span> <span class=pl-c1>lambda</span>:<span class=pl-s1>lambda</span><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1466" class="blob-num js-line-number" data-line-number="1466"></td>
        <td id="LC1466" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1467" class="blob-num js-line-number" data-line-number="1467"></td>
        <td id="LC1467" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1468" class="blob-num js-line-number" data-line-number="1468"></td>
        <td id="LC1468" class="blob-code blob-code-inner js-file-line"><span class=pl-c>// http://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript</span></td>
      </tr>
      <tr>
        <td id="L1469" class="blob-num js-line-number" data-line-number="1469"></td>
        <td id="LC1469" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>numberWithCommas</span><span class=pl-kos>(</span><span class=pl-s1>x</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1470" class="blob-num js-line-number" data-line-number="1470"></td>
        <td id="LC1470" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>parts</span> <span class=pl-c1>=</span> <span class=pl-s1>x</span><span class=pl-kos>.</span><span class=pl-en>toString</span><span class=pl-kos>(</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>split</span><span class=pl-kos>(</span><span class=pl-s>&quot;.&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1471" class="blob-num js-line-number" data-line-number="1471"></td>
        <td id="LC1471" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>parts</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span> <span class=pl-c1>=</span> <span class=pl-s1>parts</span><span class=pl-kos>[</span><span class=pl-c1>0</span><span class=pl-kos>]</span><span class=pl-kos>.</span><span class=pl-en>replace</span><span class=pl-kos>(</span><span class=pl-pds>/<span class=pl-cce>\B</span><span class=pl-kos>(?</span><span class=pl-c1>=</span><span class=pl-kos>(</span><span class=pl-cce>\d</span><span class=pl-kos>{</span>3<span class=pl-kos>}</span><span class=pl-kos>)</span><span class=pl-c1>+</span><span class=pl-kos>(?</span><span class=pl-c1>!</span><span class=pl-cce>\d</span><span class=pl-kos>)</span><span class=pl-kos>)</span>/g</span><span class=pl-kos>,</span> <span class=pl-s>&quot;,&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1472" class="blob-num js-line-number" data-line-number="1472"></td>
        <td id="LC1472" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>parts</span><span class=pl-kos>.</span><span class=pl-en>join</span><span class=pl-kos>(</span><span class=pl-s>&quot;.&quot;</span><span class=pl-kos>)</span><span class=pl-kos>;</span></td>
      </tr>
      <tr>
        <td id="L1473" class="blob-num js-line-number" data-line-number="1473"></td>
        <td id="LC1473" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
      <tr>
        <td id="L1474" class="blob-num js-line-number" data-line-number="1474"></td>
        <td id="LC1474" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1475" class="blob-num js-line-number" data-line-number="1475"></td>
        <td id="LC1475" class="blob-code blob-code-inner js-file-line"><span class=pl-k>function</span> <span class=pl-en>drawAnnotations</span><span class=pl-kos>(</span><span class=pl-s1>figure</span><span class=pl-kos>,</span> <span class=pl-s1>annotations</span><span class=pl-kos>)</span> <span class=pl-kos>{</span></td>
      </tr>
      <tr>
        <td id="L1476" class="blob-num js-line-number" data-line-number="1476"></td>
        <td id="LC1476" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1477" class="blob-num js-line-number" data-line-number="1477"></td>
        <td id="LC1477" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>figwidth</span> <span class=pl-c1>=</span> <span class=pl-s1>figure</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1478" class="blob-num js-line-number" data-line-number="1478"></td>
        <td id="LC1478" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>figheight</span> <span class=pl-c1>=</span> <span class=pl-s1>figure</span><span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1479" class="blob-num js-line-number" data-line-number="1479"></td>
        <td id="LC1479" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1480" class="blob-num js-line-number" data-line-number="1480"></td>
        <td id="LC1480" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>svg</span> <span class=pl-c1>=</span> <span class=pl-s1>figure</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;svg&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1481" class="blob-num js-line-number" data-line-number="1481"></td>
        <td id="LC1481" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;width&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>figwidth</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1482" class="blob-num js-line-number" data-line-number="1482"></td>
        <td id="LC1482" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;height&quot;</span><span class=pl-kos>,</span> <span class=pl-s1>figheight</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1483" class="blob-num js-line-number" data-line-number="1483"></td>
        <td id="LC1483" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;position&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;absolute&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1484" class="blob-num js-line-number" data-line-number="1484"></td>
        <td id="LC1484" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;top&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;0px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1485" class="blob-num js-line-number" data-line-number="1485"></td>
        <td id="LC1485" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;left&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;0px&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1486" class="blob-num js-line-number" data-line-number="1486"></td>
        <td id="LC1486" class="blob-code blob-code-inner js-file-line">                <span class=pl-kos>.</span><span class=pl-en>style</span><span class=pl-kos>(</span><span class=pl-s>&quot;pointer-events&quot;</span><span class=pl-kos>,</span><span class=pl-s>&quot;none&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1487" class="blob-num js-line-number" data-line-number="1487"></td>
        <td id="LC1487" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1488" class="blob-num js-line-number" data-line-number="1488"></td>
        <td id="LC1488" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>swoopy</span> <span class=pl-c1>=</span> <span class=pl-s1>d3</span><span class=pl-kos>.</span><span class=pl-en>swoopyDrag</span><span class=pl-kos>(</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1489" class="blob-num js-line-number" data-line-number="1489"></td>
        <td id="LC1489" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>x</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>x</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1490" class="blob-num js-line-number" data-line-number="1490"></td>
        <td id="LC1490" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>y</span><span class=pl-kos>(</span><span class=pl-k>function</span><span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>)</span><span class=pl-kos>{</span> <span class=pl-k>return</span> <span class=pl-kos>(</span><span class=pl-s1>d</span><span class=pl-kos>.</span><span class=pl-c1>y</span><span class=pl-kos>)</span> <span class=pl-kos>}</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1491" class="blob-num js-line-number" data-line-number="1491"></td>
        <td id="LC1491" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>draggable</span><span class=pl-kos>(</span><span class=pl-c1>false</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1492" class="blob-num js-line-number" data-line-number="1492"></td>
        <td id="LC1492" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>annotations</span><span class=pl-kos>(</span><span class=pl-s1>annotations</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1493" class="blob-num js-line-number" data-line-number="1493"></td>
        <td id="LC1493" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1494" class="blob-num js-line-number" data-line-number="1494"></td>
        <td id="LC1494" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>var</span> <span class=pl-s1>swoopySel</span> <span class=pl-c1>=</span> <span class=pl-s1>svg</span><span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&quot;g&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;class&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;annotatetext&quot;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>call</span><span class=pl-kos>(</span><span class=pl-s1>swoopy</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1495" class="blob-num js-line-number" data-line-number="1495"></td>
        <td id="LC1495" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1496" class="blob-num js-line-number" data-line-number="1496"></td>
        <td id="LC1496" class="blob-code blob-code-inner js-file-line">    svg.append(&#39;marker&#39;)</td>
      </tr>
      <tr>
        <td id="L1497" class="blob-num js-line-number" data-line-number="1497"></td>
        <td id="LC1497" class="blob-code blob-code-inner js-file-line">        .<span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;id&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;arrow&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1498" class="blob-num js-line-number" data-line-number="1498"></td>
        <td id="LC1498" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;viewBox&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;-10 -10 20 20&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1499" class="blob-num js-line-number" data-line-number="1499"></td>
        <td id="LC1499" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;markerWidth&#39;</span><span class=pl-kos>,</span> <span class=pl-c1>20</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1500" class="blob-num js-line-number" data-line-number="1500"></td>
        <td id="LC1500" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;markerHeight&#39;</span><span class=pl-kos>,</span> <span class=pl-c1>20</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1501" class="blob-num js-line-number" data-line-number="1501"></td>
        <td id="LC1501" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;orient&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;auto&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1502" class="blob-num js-line-number" data-line-number="1502"></td>
        <td id="LC1502" class="blob-code blob-code-inner js-file-line">      <span class=pl-kos>.</span><span class=pl-en>append</span><span class=pl-kos>(</span><span class=pl-s>&#39;path&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1503" class="blob-num js-line-number" data-line-number="1503"></td>
        <td id="LC1503" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;d&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;M-6.75,-6.75 L 0,0 L -6.75,6.75&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1504" class="blob-num js-line-number" data-line-number="1504"></td>
        <td id="LC1504" class="blob-code blob-code-inner js-file-line">        <span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&quot;transform&quot;</span><span class=pl-kos>,</span> <span class=pl-s>&quot;scale(0.5)&quot;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1505" class="blob-num js-line-number" data-line-number="1505"></td>
        <td id="LC1505" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1506" class="blob-num js-line-number" data-line-number="1506"></td>
        <td id="LC1506" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>swoopySel</span><span class=pl-kos>.</span><span class=pl-en>selectAll</span><span class=pl-kos>(</span><span class=pl-s>&#39;path&#39;</span><span class=pl-kos>)</span><span class=pl-kos>.</span><span class=pl-en>attr</span><span class=pl-kos>(</span><span class=pl-s>&#39;marker-end&#39;</span><span class=pl-kos>,</span> <span class=pl-s>&#39;url(#arrow)&#39;</span><span class=pl-kos>)</span></td>
      </tr>
      <tr>
        <td id="L1507" class="blob-num js-line-number" data-line-number="1507"></td>
        <td id="LC1507" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1508" class="blob-num js-line-number" data-line-number="1508"></td>
        <td id="LC1508" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>swoopySel</span></td>
      </tr>
      <tr>
        <td id="L1509" class="blob-num js-line-number" data-line-number="1509"></td>
        <td id="LC1509" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1510" class="blob-num js-line-number" data-line-number="1510"></td>
        <td id="LC1510" class="blob-code blob-code-inner js-file-line"><span class=pl-kos>}</span></td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 13 16" version="1.1" width="13" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm5 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM13 7.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z"/></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2" style="width:185px">
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;">
            Copy lines
          </clipboard-copy>
        </li>
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;">
            Copy permalink
          </clipboard-copy>
        </li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/distillpub/post--momentum/blame/691048b9d00b4b49b830c602b970755781df332c/public/assets/utils.js">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" role="menuitem" href="/distillpub/post--momentum/issues/new">Reference in new issue</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  

  <details class="details-reset details-overlay details-overlay-dark">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get">
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>



  </div>
</div>

    </main>
  </div>
  

  </div>

        
<div class="footer container-lg width-full p-responsive" role="contentinfo">
  <div class="position-relative d-flex flex-row-reverse flex-lg-row flex-wrap flex-lg-nowrap flex-justify-center flex-lg-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
      <li class="mr-3 mr-lg-0">&copy; 2020 GitHub, Inc.</li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to security, text:security" href="https://github.com/security">Security</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>

    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon d-none d-lg-block mx-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>
    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 000 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 00.01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" async="async" integrity="sha512-WcQmT2vhcClFVOaaAJV/M+HqsJ2Gq/myvl6F3gCVBxykazXTs+i5fvxncSXwyG1CSfcrqmLFw/R/bmFYzprX2A==" type="application/javascript" id="js-conditional-compat" data-src="https://github.githubassets.com/assets/compat-bootstrap-59c4264f.js"></script>
    <script crossorigin="anonymous" integrity="sha512-6XBdUZGib4aqdruJTnLMOLpIh0VJsGlgQ7M3vndWJIH6YQNv+zqpo1TbCDzjHJ+YYEm4xkEinaY0VsemDUfi9A==" type="application/javascript" src="https://github.githubassets.com/assets/environment-bootstrap-e9705d51.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-8aoMrIRCQt5Ybuay957ZPDJZxAgMiOWYFrdTd5MPV8LYgHGHMVUXTXKPM0yJdgOJqtChru2E3edXCTzDX9KgJg==" type="application/javascript" src="https://github.githubassets.com/assets/vendor-f1aa0cac.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-CcKFBqQZKOCZU5otP6R8GH2k+iJ3zC9r2z2Iakfs/Bo9/ptHy6JIWQN3FPhVuS3CR+Q/CkEOSfg+WJfoq3YMxQ==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-09c28506.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-j2MnK6zLa5/S0PltM8vYj6XnjOEaQDFVy9txsu7zkKjoyWA5Yus9rIvgrjFgp3gWRtSdIzDedpOOFChMDHYzkQ==" type="application/javascript" src="https://github.githubassets.com/assets/github-bootstrap-8f63272b.js"></script>
    
    
    
  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 000 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 00.01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>

  <div aria-live="polite" class="js-global-screen-reader-notice sr-only"></div>

  </body>
</html>

